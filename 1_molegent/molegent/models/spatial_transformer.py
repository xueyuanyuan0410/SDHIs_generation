import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from torch.utils.data import DataLoader

import numpy as np
from molegent.atom_alphabet import Atoms
from molegent.datasets.ase_dataset import get_ase_train_and_test_set, input_to_sequence
from molegent.molutils import MultiDimensionalScaler
from utils import PositionalEncoding, PadCollate, create_source_mask


class SpatialTransformer(nn.Module):
    """
    Transformer capable of processing molecules represented by atoms and the corresponding euclidean distance matrix.
    """

    def __init__(
        self,
        config,
        device,
    ):

        super(SpatialTransformer, self).__init__()

        self.config = config
        self.device = device

        self.atom_vocab_size = len(list(Atoms))

        # the max distance is divided into bins depending on the bin size. The transformer will predict a probability
        # distribution for the distance bins
        self.nr_bins = int(config["max_distance"] / config["bin_size"])

        embedding_dim = config["embedding_dim"]
        atom_embedding_dim = embedding_dim - 1

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=config["num_heads"], dropout=config["dropout"]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=config["num_enc_layer"])

        self.positional_encoding = PositionalEncoding(max_nr_atoms=config["max_num_atoms"], d_model=embedding_dim)

        self.atom_type_embedding = nn.Embedding(num_embeddings=self.atom_vocab_size, embedding_dim=atom_embedding_dim)

        self.embedding = nn.Linear(in_features=575, out_features=embedding_dim)

        # the atom and distance decoder are stacked linear layers
        self.decoder_atom_type = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=self.atom_vocab_size),
        )
        self.decoder_position = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.nr_bins),
        )

        self.source_mask = None

        # Multi dimensional scaling is used to fit the generated euclidean distance matrix to 3D coordinates
        #转化成坐标
        self.multi_dimensional_scaler = MultiDimensionalScaler()

    def forward(self, x):

        atom_seq = x["atoms"].to(self.device)
        distance_seq = x["connections"].to(self.device)

        # embed the atoms with a learnable embedding [N, A] -> [N, A, E_ATOM]
        atoms_emb = self.atom_type_embedding(atom_seq)

        # combine with the distance [N, A, E]
        merged_input = torch.cat([atoms_emb, distance_seq.unsqueeze(dim=2)], dim=2)

        # add an positional encoding
        emb = self.positional_encoding(merged_input)

        # create the autoregressive mask
        mask = self.get_src_mask(size=atoms_emb.shape[1], device=atoms_emb.device)
        # permute to [A, N, E]
        emb = emb.permute(1, 0, 2)
        # Encode with the Transformer
        encoded = self.encoder(src=emb, mask=mask)
        # permute back [N, A, E]
        encoded = encoded.permute(1, 0, 2)

        # predict the atoms
        predicted_atoms = self.decoder_atom_type(encoded)
        # predict the distance bin
        predicted_positions = self.decoder_position(encoded)
        print(' predicted_positions in spatial_transformer.py ')
        print(predicted_positions)

        return predicted_atoms, predicted_positions

    def sample(self, num_samples, device):
        """
        Use the model to sample new molecules.
        """

        with torch.no_grad():

            self.eval()

            max_num_atoms = self.config["max_num_atoms"]
            atom_temp = self.config.get("atom_temp", 1.0)
            distance_temp = self.config.get("distance_temp", 1.0)

            max_len = sum(range(max_num_atoms))

            # placeholder for the atom and distance input sequences
            atoms = torch.full(
                (num_samples, max_len),
                fill_value=Atoms.EOM.value,
                device=device,
                dtype=torch.int64,
            )
            distances = torch.full((num_samples, max_len), fill_value=0, device=device, dtype=torch.float32)

            # placeholder for the molecule atom list and the EDM
            mol_atoms = torch.full((num_samples, max_num_atoms), fill_value=Atoms.EOM.value)
            mol_edm = torch.zeros((num_samples, max_num_atoms, max_num_atoms))

            row_idx = 0
            col_idx = 0

            atoms[:, 0] = Atoms.BOM.value

            cur_atoms = None
            start_idx = 0

            predefined_atoms_file = self.config.get("predefined_atoms_file")
            if predefined_atoms_file is not None:
                # start with some already defined atoms and their distances
                predefined = np.loadtxt(predefined_atoms_file, dtype=np.str)
                predefined_atoms = predefined[0]
                predefined_atoms = np.array([Atoms[i].value for i in predefined_atoms])
                predefined_distances = predefined[1:].astype(np.float)

                nr_already_present_atoms = len(predefined_atoms)

                mol_atoms[:, :nr_already_present_atoms] = torch.tensor(predefined_atoms)
                mol_edm[:, :nr_already_present_atoms, :nr_already_present_atoms] = torch.tensor(predefined_distances)

                row_idx = 0
                col_idx = nr_already_present_atoms

                inputs = input_to_sequence(atoms=predefined_atoms, edm=predefined_distances)
                size = inputs["src_atoms"].shape[0]
                atoms[:, :size] = inputs["src_atoms"]
                distances[:, :size] = inputs["src_connections"]
                start_idx = size - 1

            # sample new atoms loop
            for i in range(start_idx, max_len - 1):

                pre_atoms, pre_distances = self.forward(
                    {"atoms": atoms[:, : i + 1], "connections": distances[:, : i + 1]}
                )

                atoms_logits = pre_atoms[:, i]
                atoms_logits[:, Atoms.BOM.value] = float("-inf")
                atoms_logits[:, Atoms.PAD.value] = float("-inf")

                if i == 0:
                    # set the probability for an EOM token for the first sampling step to 0
                    atoms_logits[:, Atoms.EOM.value] = float("-inf")

                if row_idx == 0:
                    # if a new row of the edm is reached sample a new atom type
                    atom_probs = F.softmax(atoms_logits / atom_temp, dim=1)
                    tokens = Categorical(probs=atom_probs).sample()
                    mol_atoms[:, col_idx] = tokens

                    cur_atoms = tokens
                    mol_atoms[:, col_idx] = cur_atoms

                atoms[:, i + 1] = cur_atoms

                if row_idx != 0:
                    # sample the next EDM entry
                    distances_logits = pre_distances[:, i]
                    distance_probs = F.softmax(distances_logits / distance_temp, dim=1)

                    # the index of the distance bin
                    distance_bin_idx = Categorical(probs=distance_probs).sample()

                    # convert the distance bin to a distance value
                    distance = distance_bin_idx * self.config["bin_size"]
                    distances[:, i + 1] = distance

                    mol_edm[:, col_idx, row_idx - 1] = distance

                if row_idx == col_idx:
                    col_idx += 1
                    row_idx = 0
                else:
                    row_idx += 1

            return [{"atoms": mol_atoms[i].cpu(), "edm": mol_edm[i].cpu()} for i in range(num_samples)]

    def calculate_loss(self, prediction, target):

        predicted_atoms = prediction[0]
        predicted_distances = prediction[1]

        tgt_atoms = target["tgt_atoms"].to(self.device)
        tgt_distances = target["tgt_connections"].to(self.device)

        # calculate the atom prediction loss
        predicted_atoms = predicted_atoms.permute([0, 2, 1])
        atom_loss = F.cross_entropy(predicted_atoms, tgt_atoms, reduction="none")

        # calculate the distance prediction loss
        target_distances_bins = torch.floor(tgt_distances / self.config["bin_size"]).to(torch.long)
        predicted_distances = predicted_distances.permute([0, 2, 1])
        distance_losses = F.cross_entropy(predicted_distances, target_distances_bins, reduction="none")

        # ignore those loss values referring to a PAD token
        padding_mask = tgt_atoms == Atoms.PAD.value
        loss_atom = torch.masked_select(atom_loss, torch.logical_not(padding_mask))
        loss_distance = torch.masked_select(distance_losses, torch.logical_not(padding_mask))

        loss = loss_atom + loss_distance
        loss = loss.mean()
        loss_atom = loss_atom.mean()
        loss_distance = loss_distance.mean()

        return {"loss": loss, "loss_atom": loss_atom, "loss_distance": loss_distance}

    def get_data_loader(self):
        """
        prepare the dataset and data loader for the spatial transformer. Molecules are sampled from the QM9 database.

        """
        max_size_train = self.config.get("max_size_train_set", None)
        max_size_test = self.config.get("max_size_test_set", None)

        train_set, test_set = get_ase_train_and_test_set(
            ase_path=self.config["qm9_path"],
            invalid_path=self.config["invalid_molecules_path"],
            max_size_test=max_size_test,
            max_size_train=max_size_train,
            shuffle_atoms=self.config["shuffle_atoms"],
        )

        collator = PadCollate(pad_token_atom=0, pad_value_position=0)

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_data_load_worker"],
            collate_fn=collator,
            shuffle=self.config["shuffle_samples"],
        )

        test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_data_load_worker"],
            collate_fn=collator,
            shuffle=self.config["shuffle_samples"],
        )

        return train_set, test_set, train_loader, test_loader

    def get_src_mask(self, size: int, device="cpu"):
        """
        create an autoregressive mask for the given size, or return a fitting cached mask
        """
        if self.source_mask is None or self.source_mask.shape[0] != size:
            self.source_mask = create_source_mask(size, device)

        return self.source_mask
