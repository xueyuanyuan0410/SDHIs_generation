import logging
import random

import numpy as np
import torch
from ase.db import connect
from scipy import spatial
from torch.utils.data import Dataset

from molegent.atom_alphabet import Atoms


class ASE_Dataset(Dataset):
    def __init__(self, ase_db_path, indices_of_samples, shuffle_atoms=False):

        super(ASE_Dataset, self).__init__()

        self.indices_of_samples = indices_of_samples
        self.db_path = ase_db_path
        self.shuffle_atoms = shuffle_atoms

        self.indices = indices_of_samples

    def __getitem__(self, index):

        # sample a molecule, represented by atoms and their coordinates (x, y, z)
        atoms, positions = self._get_atoms_and_positions(index)
        assert len(atoms) > 0

        if self.shuffle_atoms:
            indices = list(range(len(atoms)))
            random.shuffle(indices)
            atoms = [atoms[i] for i in indices]
            positions = np.array([positions[i] for i in indices])

        # calculate target distributions of atom types
        # Add the BOM and EOM token to the list of atoms
        atoms = np.array(atoms)

        # calculate the euclidean distance for between every two atoms, to get the euclidean distance matrix
        edm = spatial.distance.cdist(positions, positions)

        inputs_sequences = input_to_sequence(atoms=atoms, edm=edm)

        return inputs_sequences

    def _get_atoms_and_positions(self, sample_index):

        db_idx = self.indices[sample_index]

        with connect(self.db_path) as db:
            row = db.get(db_idx)
            atoms_obj = db.get_atoms(db_idx)

            not_h = [i for i in range(len(row.symbols)) if row.symbols[i] != "H"]
            atoms = [Atoms[row.symbols[i]].value for i in not_h]

            positions = row.positions[not_h]

            atoms, positions = self.heuristic_sort(atoms_obj, atoms, positions)

            return atoms, positions

    def heuristic_sort(self, atoms_obj, atoms, positions):
        com = atoms_obj.get_center_of_mass()
        com = np.expand_dims(com, axis=0)
        distances = spatial.distance.cdist(positions, com)
        sorted_indices = np.argsort(distances.squeeze())

        return np.array(atoms)[sorted_indices], positions[sorted_indices]

    def __len__(self):
        return len(self.indices)


def input_to_sequence(atoms, edm):
    # only keep the upper right triangle
    edm = torch.tensor(np.triu(edm), dtype=torch.float32).T
    mask = torch.triu(torch.ones_like(edm)).to(torch.bool).T

    distances = torch.masked_select(edm, mask)
    atoms_stacked = torch.tensor(np.repeat(atoms.reshape(1, -1), repeats=edm.shape[0], axis=0), dtype=torch.long).T
    tgt_atoms = torch.roll(atoms_stacked, shifts=-1, dims=0)
    tgt_atoms[-1, :] = Atoms.EOM.value

    src_atom_seq = torch.masked_select(atoms_stacked, mask)
    tgt_atom_seq = torch.masked_select(tgt_atoms, mask)

    src_atom_seq = torch.cat((torch.tensor([Atoms.BOM.value]), src_atom_seq))
    tgt_atom_seq = torch.cat((atoms_stacked[0, 0].unsqueeze(dim=0), tgt_atom_seq))

    tgt_distance_seq = torch.cat((torch.tensor([0]), distances))
    src_distance_seq = torch.cat((torch.tensor([0, 0]), distances[:-1]))

    return {
        "src_atoms": src_atom_seq,
        "tgt_atoms": tgt_atom_seq,
        "src_connections": src_distance_seq,
        "tgt_connections": tgt_distance_seq,
    }


def get_ase_train_and_test_set(
    ase_path,
    invalid_path=None,
    shuffle_atoms=False,
    max_size_train=None,
    max_size_test=None,
    train_test_split=0.9,
    random_seed=None,
):

    with connect(ase_path) as db:
        db_len = db.count()
        indices = list(range(1, db_len + 1))

    if invalid_path:
        with open(invalid_path, "r") as invalid_file:
            invalid = [int(l) for l in invalid_file]

            indices = list(set(indices) - set(invalid))

    if random_seed:
        logging.getLogger("main").info(f"Data set random seed: {random_seed}")
        random.seed(random_seed)

    random.shuffle(indices)

    split = int(train_test_split * len(indices))
    train_indices = indices[:split]
    test_indices = indices[split:]

    if max_size_train:
        train_indices = train_indices[:max_size_train]
    if max_size_test:
        test_indices = test_indices[:max_size_test]

    train_dataset = ASE_Dataset(
        ase_db_path=ase_path,
        indices_of_samples=train_indices,
        shuffle_atoms=shuffle_atoms,
    )
    test_dataset = ASE_Dataset(
        ase_db_path=ase_path,
        indices_of_samples=test_indices,
        shuffle_atoms=shuffle_atoms,
    )

    return train_dataset, test_dataset
