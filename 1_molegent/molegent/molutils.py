import os
import random

from openbabel import openbabel as ob
import moses
import numpy as np
import openbabel.pybel as pb
import pandas
from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromPDBBlock #Construct a molecule from a PDB block.
from sklearn.manifold import MDS

from atom_alphabet import Atoms, ZincAtoms
from molecule import Molecule
import pandas as pd
from rdkit.Chem import AllChem

BOND_TYPES = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE} 

class MolAnalyzer:
    """
    uses a MolConstructor to construct the molecules and analyzes them with different metrics 
    """

    def __init__(self):

        self.mol_constructor = MolConstructor()

    def analyze_mol_tensor(self, sampled_info, output_dir):

        # construct the molecules. Each molecule is defined by a list of atoms and their distances
        mols = self.mol_constructor.construct_mols(sampled_info)

        valid_mols = [Molecule(x) for x in mols if x is not None]

        metrics = self.calculate_metrics(valid_mols)

        fully_connected = [(m["fully_connected"]) for m in metrics]

        valid_mols_rdkit = []
        for mol in mols:
            if mol is not None:
                valid_mols_rdkit.append(mol)

        full_mols = []
        for i, _ in enumerate(valid_mols_rdkit):
            if fully_connected[i] == True:
                full_mols.append(valid_mols_rdkit[i])

        mol_dict = []
        for i in full_mols:
            mol_dict.append({'molecule': i, 'smiles': Chem.MolToSmiles(i)})

        results = pd.DataFrame(mol_dict)
        results.to_csv('/home/fungicide_generation/molegent_C2/molegent/results/with_data_aug_genetating_C2.csv', index=False,mode='w')

        nr_valid = fully_connected.count(True) / len(sampled_info) * 100
        nr_wrong_valence = (len(mols) - len(valid_mols)) / len(mols) * 100
        nr_not_connected = fully_connected.count(False) / len(mols) * 100

        if output_dir:
            self.save_results(
                output_dir,
                metrics=metrics,
                ratio_valid=nr_valid,
                ratio_not_connected=nr_not_connected,
                ratio_wrong_valence=nr_wrong_valence,
            )

        return metrics, nr_valid, nr_wrong_valence, nr_not_connected

    def save_results(
        self,
        output_dir,
        metrics,
        ratio_valid,
        ratio_not_connected,
        ratio_wrong_valence,
    ):
        os.makedirs(output_dir)
        df = pandas.DataFrame.from_dict(metrics)
        df.to_csv(f"{output_dir}/overview.csv")

        with open(f"{output_dir}/metrics.txt", "w") as f:
            f.write(f"valid: {ratio_valid}\n")
            f.write(f"not connected: {ratio_not_connected}\n")
            f.write(f"wrong valence: {ratio_wrong_valence}\n")

    def calculate_metrics(self, molecules):
        metrics = [m.get_dict_of_metrics() for m in molecules]
        return metrics


class MolConstructor:
    """
    can construct rdkit molecules from tensors of atoms and their corresponding distance matrix
    """

    def __init__(self):
        self.multi_dimensional_scaler = MultiDimensionalScaler()

    def construct_mols(self, sampled_info):

        mols = []

        for mol_info in sampled_info:
            if "edm" in mol_info:
                mol = self.construct_mols_from_distances(atoms=mol_info["atoms"], edm=mol_info["edm"])
            elif "adjacency_matrix" in mol_info:
                mol = self.construct_mol_from_graph(atoms=mol_info["atoms"], amat=mol_info["adjacency_matrix"])
            else:
                raise AttributeError("Not enough info to construct the molecule")
            mols.append(mol)

        return mols 

    def construct_mols_from_distances(self, atoms, edm):

        mols = []
        stress_values = []

        # remove BOM and EOM
        mol_atoms, mol_distances = self.trim_bom_and_eom(atoms, edm)
        if len(mol_atoms) > 1:
            # calculate the coordinates from the distance matrix
            coords, stress = self.multi_dimensional_scaler.edm_to_coords(mol_distances)
        else:
            coords = np.array([[0, 0, 0]])
            stress = 0.0
        # create a pdb-string to read in the molecule. Seems unnecessary complicated, and it is. But who knows...
        pdb_block = self._create_pdb_block_for_molecule(mol_atoms=mol_atoms, mol_coords=coords)

        # read in the molecule an satitize
        mol = MolFromPDBBlock(pdb_block, sanitize=True)

        return mol

    def _create_pdb_block_for_molecule(self, mol_atoms, mol_coords):
        """
        write a string representing a pdb file of the molecule
        """
        pdb_block = ""
        atom_count = np.zeros(len(Atoms), dtype=np.int)

        for i in range(len(mol_atoms)):
            atom = mol_atoms[i]
            atom_index = atom.value

            atom_count[atom_index] = atom_count[atom_index] + 1

            pdb_block += f"HETATM{i+1:>5}{atom.name:>3}{atom_count[atom_index]:<3}{'UNL':<8}1    {mol_coords[i, 0]:8.3f}{mol_coords[i, 1]:8.3f}{mol_coords[i, 2]:8.3f}  1.00  0.00         {atom.name:>3}  \n"

        pdb_block += "END"
        return pdb_block

    def construct_mol_from_graph(self, atoms, bonds=None, amat=None):

        mol = Chem.RWMol()

        trimmed_atoms = self.trim_bom_and_eom(atoms, atom_alphabet=ZincAtoms)

        for atom in trimmed_atoms:
            mol.AddAtom(Chem.Atom(atom.name))

        if bonds is not None:
            for bond in bonds:

                if bond[1] > len(trimmed_atoms):
                    break

                mol.AddBond(bond[0] - 1, bond[1] - 1, BOND_TYPES[1])

        elif amat is not None:
            amat = self.trim_adjacency_matrix(amat=amat, len=len(trimmed_atoms))
            for i in range(amat.shape[0]):

                for j in range(amat.shape[1]):
                    if i == j:
                        break

                    connected = amat[i, j].item()
                    if connected:
                        mol.AddBond(i, j, BOND_TYPES[connected])

        mol = mol.GetMol()
        # mol.UpdatePropertyCache()

        try:
            Chem.SanitizeMol(mol)
            return mol
        except ValueError:
            return None

    def trim_bom_and_eom(self, atoms, edm=None, atom_alphabet=Atoms):
        """
        remove the BOM and EOM tokens from the molecule and return only the real atoms and the correct distance matrix
        """

        mol_atoms = []

        for i in range(atoms.shape[0]):
            atom_index = atoms[i].item()
            atom = atom_alphabet(atom_index)

            if atom == atom_alphabet.EOM:
                break

            mol_atoms.append(atom)

        if edm is not None:
            mol_edm = edm[: len(mol_atoms), : len(mol_atoms)].numpy()
            return mol_atoms, mol_edm

        return mol_atoms

    def trim_adjacency_matrix(self, amat, len):
        return amat[:len, :len]


class MultiDimensionalScaler:
    def __init__(self):
        self.mds = MDS(n_components=3, dissimilarity="precomputed")

    def edm_to_coords(self, edm):
        # create a symmetric distance matrix
        edm = edm + edm.transpose()

        # use multi dimensional scaling to embed the distances in a 3-d space
        weights = np.ones_like(edm)
        weights[:3, :3] = 10
        weights[np.eye(weights.shape[0], dtype=np.bool)] = 0 
        # mds_results = self.mds.fit(edm, weight=None)
        mds_results = self.mds.fit(edm)

        coords = mds_results.embedding_
        # also collect the stress value for this embedding
        stress = mds_results.stress_
        return coords, stress

def construct_adjacancy_matrix(pybel_mol: pb.Molecule, default_value=0, atom_mapping=None):
    """
    construct the adjacency matrix, including bond orders, for a openbabel (pybel) molecule.
    """

    mol = pybel_mol.OBMol

    num_atoms = mol.NumAtoms()

    if atom_mapping is None:
        atom_mapping = range(num_atoms)

    amat = np.full(shape=(num_atoms, num_atoms), fill_value=default_value)

    for b in ob.OBMolBondIter(mol):
        bond_order = b.GetBondOrder()
        begin_idx = b.GetBeginAtomIdx() - 1
        end_idx = b.GetEndAtomIdx() - 1

        begin_atom = atom_mapping.index(begin_idx) 
        end_atom = atom_mapping.index(end_idx)

        amat[begin_atom, end_atom] = bond_order

    return amat + amat.T


def get_list_atom_types(pybel_mol: pb.Molecule, shuffle="no"):
    atoms = [ZincAtoms.from_atomic_number(a.atomicnum).value for a in pybel_mol.atoms]
    atom_mapping = list(range(len(atoms)))

    if shuffle == "random":
        random.shuffle(atom_mapping)
    elif shuffle == "random_df_sort":
        start_idx = random.randrange(1, len(atoms) + 1)
        atom_mapping = ob.OBMolAtomDFSIter(pybel_mol.OBMol, start_idx)
        atom_mapping = [t.GetIdx() - 1 for t in atom_mapping]
    elif shuffle != "no":
        raise AttributeError(f"unknown option chosen for the atom shuffle strategy: {shuffle}")

    atoms = [atoms[a] for a in atom_mapping]
    return atoms, atom_mapping


def get_list_bonds(pybel_mol: pb.Molecule):
    mol = pybel_mol.OBMol

    bonds = [b for b in ob.OBMolBondIter(mol)]

    edge_indices = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in bonds]).transpose([1, 0])
    edge_feature = np.array([b.GetBondOrder() for b in bonds])

    return edge_indices, edge_feature

def domain_aug(smile, row_idx, rule_indicator, rules, aug_times=10):
    mol_obj = Chem.MolFromSmiles(smile)
    aug_smiles = []
    mol_prev = mol_obj
    # mol_next = None
    global products
    for time in range(aug_times):
        print('aug time: ', time)
        non_zero_idx = list(np.where(rule_indicator[row_idx, :] != 0)[0])
        cnt = -1
        while len(non_zero_idx) != 0:
            col_idx = random.choice(non_zero_idx)

            # calculate counts
            rule = rules[col_idx]
            rxn = AllChem.ReactionFromSmarts(rule['smarts'])
            products = rxn.RunReactants((mol_prev,))

            

            cnt = len(products)
            if cnt != 0:
                break
            else:
                non_zero_idx.remove(col_idx)


        valid_mols = []
        for mol in products:
            try:
                Chem.SanitizeMol(mol[0])
            except ValueError:  # TODO: add detailed exception
                mol=None
            if mol is not None:
                valid_mols.append(mol)

        if len(valid_mols) >= 1:
            aug_idx = random.choice(range(len(valid_mols)))
            mol = valid_mols[aug_idx][0]
            # mol = products[aug_idx][0]
            # try:
            #     Chem.SanitizeMol(mol)
            # except:  # TODO: add detailed exception
            #     pass

            aug_smiles.append(Chem.MolToSmiles(mol))

            # mol_next = mol
            # mol_prev = mol
            # rule_indicator[row_idx, col_idx] -= 1
        # else:
        #     mol_next = mol_prev

    return aug_smiles
