import moses
import numpy as np
import openbabel.pybel as pb
import torch
import pandas as pd
import json
import pickle as pkl

from molegent.atom_alphabet import 

from molegent.molutils import construct_adjacancy_matrix, get_list_atom_types, domain_aug

class ZincDataset:
    def __init__(self, split: str, max_num_atoms: int, max_size: int, shuffle_strategy="no"):

        self.shuffle_strategy = shuffle_strategy
        # self._dataset = moses.get_dataset(split)
        self._max_num_atoms = max_num_atoms

        self.rules = json.load(open('/home/zy/fungicide_generation/molegent_G1/molegent/molegent/datasets/isostere_transformations_new.json'))
        with open('/home/zy/fungicide_generation/molegent_C2/molegent_C2/molegent/molegent/datasets/results/C2_rule_indicator_new.pkl', 'rb') as f:
            d = pkl.load(f)
            rule_indicator = d[0]
            print('rule indicator shape: ', rule_indicator.shape)
        self.rule_indicator = rule_indicator
    

        self._dataset = self.read_smiles(split, True) #Use data augmentation
        #self._dataset = self.read_smiles(split, False) 

        self._max_size = max_size if max_size is not None else float("inf")

    def __len__(self):
        return min(self._max_size, len(self._dataset))

    def __getitem__(self, index):
        smiles = self._dataset[index]

        mol = pb.readstring("smi", smiles)
        mol.removeh()

        atoms, atom_mapping = get_list_atom_types(mol, shuffle=self.shuffle_strategy)
        amat = construct_adjacancy_matrix(mol, atom_mapping=atom_mapping)

        src_atom_seq, src_conn_seq, tgt_atom_seq, tgt_conn_seq = self._prepare_src_tgt_sequence(atoms, amat)

        return_values = {
            "src_atoms": src_atom_seq,
            "src_connections": src_conn_seq,
            "tgt_atoms": tgt_atom_seq,
            "tgt_connections": tgt_conn_seq,
        }

        return_values["smiles"] = smiles

        return return_values

    def read_smiles(self, split, augmentation=False):
        path = "/home/fungicide_generation/molegent_C2/molegent/data/C2.csv" 
        data = pd.read_csv(path)

        if augmentation:
            aug_smiles = []
            for i, smile in enumerate(data['SMILES']):
                aug_smile = domain_aug(smile, i, self.rule_indicator, self.rules, aug_times=10)
                if (data['SPLIT'][i] == split) & (aug_smile is not None):
                        aug_smiles.extend(aug_smile)

            train_data = data[data['SPLIT'] == split].reset_index(drop=True)
            smiles = train_data['SMILES'].values.tolist()
            all_smiles = aug_smiles + smiles
        else:
            train_data = data[data['SPLIT'] == split].reset_index(drop=True)
            smiles = train_data['SMILES'].values.tolist()
            all_smiles =smiles

        return all_smiles

    def _prepare_src_tgt_sequence(self, atoms, adjacency_matrix):
        atoms = torch.tensor(atoms)
        src_atom_seq = torch.cat((torch.tensor([Atoms.BOM.value]), atoms))
        tgt_atom_seq = torch.cat((atoms, torch.tensor([Atoms.EOM.value])))

        amat = np.triu(adjacency_matrix)  #upper triangle of an array
        amat = np.pad(amat, ((0, self._max_num_atoms - amat.shape[0]), (0, 0))).T
        amat = torch.tensor(amat, dtype=torch.float32)

        src_conn_seq = torch.cat((torch.zeros((1, self._max_num_atoms)), amat))#BOM
        tgt_conn_seq = torch.cat((amat, torch.zeros((1, self._max_num_atoms))))#EOM

        return src_atom_seq, src_conn_seq, tgt_atom_seq, tgt_conn_seq

    def _get_atoms(self, mol):
        atoms = mol.GetAtoms()
        return np.array([Atoms[a.GetSymbol()].value for a in atoms])


def get_zinc_train_and_test_set(
    max_num_atoms, shuffle_strategy, max_size_train=None, max_size_test=None
):


    train_dataset = ZincDataset(
        split="train",
        max_size=max_size_train,
        max_num_atoms=max_num_atoms,
        shuffle_strategy=shuffle_strategy,
    )
    #test_scaffolds
    test_dataset = ZincDataset(
        split="test_scaffolds",
        max_size=max_size_test,
        max_num_atoms=max_num_atoms,
        shuffle_strategy=shuffle_strategy,
    )

    return train_dataset, test_dataset
