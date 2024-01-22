import torch
from torch import nn
import torch.distributions as D
from torch.utils.data import DataLoader

from molegent.atom_alphabet import ZincAtoms
from molegent.datasets.zinc_smiles_dataset import get_zinc_train_and_test_set
from utils import PadCollate, create_source_mask, PositionalEncoding, TransposeNBatchNorm #当前这个路径为什么在/home/zy/fungicide_generation/molegent_G1/molegent下
from torch.nn import functional as F

BOND_TYPES = 4
N_COMPONENTS = 20


class GraphTransformer(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.atom_vocab_size = len(list(ZincAtoms))
        self.device = device
        self.config = config
        self.atom_embedding_dim = config["atom_embedding_dim"]
        self.bond_embedding_dim = config["bond_embedding_dim"]
        self.max_num_atoms = config["max_num_atoms"]
        self._sorce_mask = None

        embedding_dim = self.atom_embedding_dim + self.bond_embedding_dim

        self.positional_encoding = PositionalEncoding(max_nr_atoms=self.max_num_atoms, d_model=embedding_dim)
        self.atom_embeddig_dim = self.atom_embedding_dim
        #vocab_size---->512
        self.atom_embedding = nn.Embedding(num_embeddings=self.atom_vocab_size, embedding_dim=self.atom_embedding_dim)
        #max_num_atoms---->512
        self.edge_embedding = nn.Linear(self.max_num_atoms, self.bond_embedding_dim)

        #embedding_dim 1024
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=config["num_heads"], dropout=config["dropout"]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=config["num_enc_layer"])#graph:num_enc_layer: 8

        # construct the atom decoder:
        atom_decoder_layers = []
        in_features = embedding_dim
        decoder_sizes = config["atom_decoder"]#graph [800, 600, 400, 200]
        for out_size in decoder_sizes:
            atom_decoder_layers.append(nn.Linear(in_features=in_features, out_features=out_size))
            atom_decoder_layers.append(TransposeNBatchNorm(out_size))
            atom_decoder_layers.append(nn.ReLU())
            in_features = out_size
        atom_decoder_layers.append(nn.Linear(in_features=in_features, out_features=self.atom_vocab_size))
        self.decoder_atom_type = nn.Sequential(*atom_decoder_layers)

        # construct the edge decoder:
        edge_decoder_layers = []
        in_features = embedding_dim + self.atom_embeddig_dim
        decoder_sizes = config["edge_decoder"]#graph [600,800,1000,2000]
        for out_size in decoder_sizes:
            edge_decoder_layers.append(nn.Linear(in_features=in_features, out_features=out_size))
            edge_decoder_layers.append(TransposeNBatchNorm(out_size))
            edge_decoder_layers.append(nn.ReLU())
            in_features = out_size
        edge_decoder_layers.append(
            nn.Linear(in_features=in_features, out_features=self.max_num_atoms * N_COMPONENTS * BOND_TYPES)
        )
        self.decoder_edges = nn.Sequential(*edge_decoder_layers)

        self.alpha_decoder = nn.Linear(in_features=embedding_dim + self.atom_embeddig_dim, out_features=N_COMPONENTS)

    def forward(self, x):

        emb = self.embedd_inputs(x)
        tgt_embedded = self.atom_embedding(x["tgt_atoms"].to(self.device))
        # encode the atom and graph sequence
        encoded = self.encode_with_transformer(embedded=emb)

        # use the encoded sequence to predict the target atoms
        atom_type_logits = self.decoder_atom_type(encoded)

        # use the ground truth target atoms and the encoded sequence to predict the bonds，这里的encoded sequence是指原子和键构成的编码吗？
        encoded = torch.cat((encoded, tgt_embedded), dim=2)

        edge_logits = self.decoder_edges(encoded)
        edge_logits = edge_logits.view(
            edge_logits.shape[0], edge_logits.shape[1], N_COMPONENTS, self.max_num_atoms, BOND_TYPES
        )#还是不大明白这个地方为什么要转化为这种形状

        # the probabilities for the mixture components
        alphas = self.alpha_decoder(encoded)

        return atom_type_logits, edge_logits, alphas

    def embedd_inputs(self, inputs):

        atom_seq, edge_seq = inputs["atoms"].to(self.device), inputs["connections"].to(self.device)
        atoms_emb = self.atom_embedding(atom_seq)
        edges_emb = self.edge_embedding(edge_seq)

        emb = torch.cat([atoms_emb, edges_emb], dim=2)

        emb = self.positional_encoding(emb)

        return emb

    def encode_with_transformer(self, embedded):

        # create the autoregressive mask
        # permute to [A, N, E]
        embedded = embedded.permute(1, 0, 2)

        if self.config["encoder_type"] == "transformer":
            mask = self._get_src_mask(size=embedded.shape[0], device=embedded.device)
            # Encode with the Transformer
            encoded = self.encoder(src=embedded, mask=mask)

        else:
            encoded = self.encoder(embedded)[0]

        # permute back [N, A, E]
        encoded = encoded.permute(1, 0, 2)

        return encoded

    def sample_next_atom_and_bonds(self, inputs):

        atom_temp = self.config.get("atom_temp", 1.0)
        alpha_temp = self.config.get("alpha_temp", 1.0)
        bond_type_temp = self.config.get("bond_type_temp", 1.0)

        emb = self.embedd_inputs(inputs)
        # encode the already sample atoms and bonds
        encoded = self.encode_with_transformer(embedded=emb)

        # get the atom type logits for the next atom
        atom_type_logits = self.decoder_atom_type(encoded)[:, -1]

        # we never want to sample BOM of PAD tokens
        atom_type_logits[:, ZincAtoms.BOM.value] = float("-inf")
        atom_type_logits[:, ZincAtoms.PAD.value] = float("-inf")

        if encoded.shape[1] == 1:
            # for the first atom set the probability fo an EOM token to 0
            atom_type_logits[:, ZincAtoms.EOM.value] = float("-inf")

        # predict the probability for the next atom
        atom_probs = F.softmax(atom_type_logits / atom_temp, dim=1)
        # sample next atom types
        sampled_atoms = D.Categorical(probs=atom_probs).sample()

        # for the bond prediction concat the encoded graph info with the information about the sampled atom
        # this will allow the model to take the sampled atom types into consideration when predicting the edges
        tgt_atoms = torch.zeros((encoded.shape[0], encoded.shape[1]), dtype=torch.int64, device=self.device)
        tgt_atoms[:, -1] = sampled_atoms
        tgt_embedded = self.atom_embedding(tgt_atoms)
        encoded = torch.cat((encoded, tgt_embedded), dim=2)

        # predict the edge logits: shape [B, C, NUM_ATOMS, MAX_NUM_ATOMS]
        # C is the number of mixture components
        edge_logits = self.decoder_edges(encoded)
        edge_logits = edge_logits.view(
            edge_logits.shape[0], edge_logits.shape[1], N_COMPONENTS, self.max_num_atoms, BOND_TYPES
        )#这个地方的维度为什么变成这样不大懂
        pred_cons_logits = edge_logits[:, -1]

        # alpha is the probability for the corresponding mixture component
        alphas = self.alpha_decoder(encoded)[:, -1]
        alpha_prob = torch.softmax(alphas / alpha_temp, dim=1)
        mix = D.Categorical(probs=alpha_prob)

        # sample the next column of the adjacency matrix
        pred_cons_probs = torch.softmax(pred_cons_logits / bond_type_temp, dim=3)
        comp = D.Independent(D.Categorical(probs=pred_cons_probs), 1)
        cmm = D.MixtureSameFamily(mix, comp)
        bonds = cmm.sample()

        return sampled_atoms, bonds

    def _get_src_mask(self, size: int, device="cpu"):
        """
        create an autoregressive mask for the given size, or return a fitting cached mask
        """
        if self._sorce_mask is None or self._sorce_mask.shape[0] != size:
            self._sorce_mask = create_source_mask(size, device)

        return self._sorce_mask

    def sample(self, num_samples, device):
        with torch.no_grad():

            self.eval()

            # atom and bond infos will be stored here
            atoms = torch.full(
                (num_samples, self.max_num_atoms), fill_value=ZincAtoms.EOM.value, device=device, dtype=torch.int64
            )
            connections = torch.full(
                (num_samples, self.max_num_atoms, self.max_num_atoms), fill_value=0, device=device, dtype=torch.float32
            )

            # start every molecule with an BOM token
            atoms[:, 0] = ZincAtoms.BOM.value

            for i in range(self.max_num_atoms - 1):

                # sample the next atom and column of the adjacency matrix
                tokens, bonds = self.sample_next_atom_and_bonds(
                    {"atoms": atoms[:, : i + 1], "connections": connections[:, : i + 1, :]}
                )

                # store and use the sampled information in the next sampling step
                atoms[:, i + 1] = tokens
                connections[:, i + 1, : i + 1] = bonds[:, : i + 1]

            return [
                {"atoms": atoms[i, 1:].cpu(), "adjacency_matrix": connections[i, 1:].cpu()} for i in range(num_samples)
            ]

    def calculate_loss(self, prediction, target):

        predicted_atoms = prediction[0]
        predicted_cons = prediction[1]
        alphas = prediction[2]
        tgt_atoms = target["tgt_atoms"].to(self.device)
        tgt_bonds = target["tgt_connections"].to(self.device)

        # calculate the atom prediction loss
        predicted_atoms = predicted_atoms.permute([0, 2, 1])
        loss_atom = F.cross_entropy(predicted_atoms, tgt_atoms, reduction="none")

        bond_losses = []
        for atom_idx in range(predicted_cons.shape[1]):

            # select all logits for the current atom idx [B, ATOM_TYPES, NUM_COMPONENTS, MAX_A]
            logits = predicted_cons[:, atom_idx]

            # only select those logits corresponding to the target atom at atom idx
            # tgt_atom = tgt_atoms[:, atom_idx].view(-1, 1, 1).expand_as(logits[:, 0]).unsqueeze(dim=1)

            # logits = torch.gather(logits, index=tgt_atom, dim=2).squeeze()

            alpha = alphas[:, atom_idx]    #mixture components的概率
            tgt_for_atom = tgt_bonds[:, atom_idx]

            mix = D.Categorical(logits=alpha)  #根据概率来产生指定shape的分布

            comp = D.Independent(D.Categorical(logits=logits), 1)
            cmm = D.MixtureSameFamily(mix, comp)   #构造独立分佈组成的混合模型
            logprob = cmm.log_prob(tgt_for_atom)
            bond_losses.append(-logprob)
        loss_bond = torch.stack(bond_losses, dim=1)

        padding_mask = tgt_atoms == ZincAtoms.PAD.value

        # select the distances losses depending on the mask
        loss_atom = torch.masked_select(loss_atom, torch.logical_not(padding_mask))#torch.logical_not()是对单个张量对象执行的。如果该值为假或0，则返回True，如果该值为True或不等于0，则返回False，它需要一个张量作为参数。

        loss_bond = torch.masked_select(loss_bond, torch.logical_not(padding_mask))

        loss = loss_atom + loss_bond
        loss = loss.mean()
        loss_atom = loss_atom.mean()
        loss_bond = loss_bond.mean()

        return {"loss": loss, "loss_atom": loss_atom, "loss_bond": loss_bond}

    def get_data_loader(self):

        max_size_train = self.config["max_size_trainset"] if "max_size_trainset" in self.config else None
        max_size_test = self.config["max_size_testset"] if "max_size_testset" in self.config else None

        #分训练集和测试集
        trainset, testset = get_zinc_train_and_test_set(
            max_size_train=max_size_train,
            max_size_test=max_size_test,
            max_num_atoms=self.config["max_num_atoms"],
            shuffle_strategy=self.config["atom_shuffle_strategy"],
        )

        graphcollater = PadCollate(pad_token_atom=0, pad_value_position=0)#因为定义了__call__，补齐操作

        trainloader = DataLoader(
            dataset=trainset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_data_load_worker"],
            collate_fn=graphcollater,
            shuffle=self.config["shuffle_samples"],
        )

        testloader = DataLoader(
            dataset=testset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_data_load_worker"],
            collate_fn=graphcollater,
            shuffle=self.config["shuffle_samples"],
        )

        return trainset, testset, trainloader, testloader
