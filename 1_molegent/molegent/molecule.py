import names
from rdkit import Chem
from datetime import datetime
from rdkit.Chem.rdMolTransforms import GetBondLength


class Molecule:
    def __init__(self, mol, named=True):

        self._mol = mol

        self._name = None
        if named:
            time = datetime.now().strftime("%y%d%m-%H%M%S%f")
            self._name = f"mol-{time}-{names.get_first_name()}"

        self._smiles = None
        self._ring_sizes = None
        self._atoms_as_symbols = None

        self._cc_distances = None
        self._co_distances = None
        self._ccc_angles = None
        self._cco_angles = None

    @property
    def smiles(self):

        if self._smiles is None:
            self._smiles = Chem.MolToSmiles(
                self._mol,
            )

        return self._smiles

    @property
    def inchi(self):
        return Chem.MolToInchi(self._mol)

    @property
    def num_atoms(self):
        return len(self._mol.GetAtoms())

    @property
    def num_rings(self):
        return self._mol.GetRingInfo().NumRings()

    @property
    def ring_sizes(self):
        if self._ring_sizes is None:
            self._ring_sizes = [len(ring) for ring in self._mol.GetRingInfo().AtomRings()]

        return self._ring_sizes

    @property
    def atoms_as_symbols(self):
        if self._atoms_as_symbols is None:
            self._atoms_as_symbols = [a.GetSymbol() for a in self._mol.GetAtoms()]

        return self._atoms_as_symbols

    @property
    def fully_connected(self):
        return not "." in self._smiles

    @property
    def distances_and_angles(self):
        """
        Calculate and return the distances between the c-c atom pairs and c-o atom pairs, as well as the angles
        between c-c-c and c-c-o chains. These values can be used to compare a set of molecules in this structural
        properties.
        :return cc_distances, co_distances, ccc_angles, cco_angles
        """
        if self._cc_distances is not None:
            return self._cc_distances, self._co_distances

        self._cc_distances = []
        self._co_distances = []
        self._ccc_angles = []
        self._cco_angles = []

        conformer = self._mol.GetConformer()

        mol = Chem.RemoveHs(self._mol)

        for atom_idx in range(mol.GetNumAtoms()):

            # calculate distances between cc and co pairs
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_symbol = atom.GetSymbol()

            if atom_symbol == "C" or atom_symbol == "O":
                for other_atom_idx in range(atom_idx + 1, mol.GetNumAtoms()):
                    other_atom = mol.GetAtomWithIdx(other_atom_idx)
                    other_atom_symbol = other_atom.GetSymbol()

                    if other_atom_symbol == "O" and atom_symbol is "C":
                        self._co_distances.append(GetBondLength(conformer, atom_idx, other_atom_idx))

                    if other_atom_symbol == "C" and atom_symbol is "C":
                        self._cc_distances.append(GetBondLength(conformer, atom_idx, other_atom_idx))

                    if other_atom_symbol == "C" and atom_symbol is "O":
                        self._co_distances.append(GetBondLength(conformer, atom_idx, other_atom_idx))

            # calculate angles between ccc and cco chains
            if atom_symbol == "C":
                neighs = atom.GetNeighbors()

                for neigh_idx, neigh_atom in enumerate(neighs):
                    symb1 = neigh_atom.GetSymbol()
                    for neigh2_idx, neigh2_atom in enumerate(neighs[neigh_idx + 1 :]):
                        symb2 = neigh2_atom.GetSymbol()

                        atom_symbols = set((atom_symbol, symb1, symb2))

                        if atom_symbols == set(("C", "C", "O")):
                            angle = Chem.rdMolTransforms.GetAngleDeg(
                                conformer, neigh_atom.GetIdx(), atom_idx, neigh2_atom.GetIdx()
                            )
                            self._cco_angles.append(angle)

                        if atom_symbols == set(("C", "C", "C")):
                            angle = Chem.rdMolTransforms.GetAngleDeg(
                                conformer, neigh_atom.GetIdx(), atom_idx, neigh2_atom.GetIdx()
                            )
                            self._ccc_angles.append(angle)

        return self._cc_distances, self._co_distances, self._ccc_angles, self._cco_angles

    def save_as_xyz(self, base_dir, save_although_unconnected=False):
        if save_although_unconnected or self.fully_connected:
            mol = Chem.AddHs(self._mol, addCoords=True)
            Chem.MolToXYZFile(mol, filename=f"{base_dir}/{self._name}.xyz")

    def get_dict_of_metrics(self):
        metrics = {
            "smiles": self.smiles,
            "inchi": self.inchi,
            "nr_atoms": self.num_atoms,
            "nr_rings": self.num_rings,
            "nr_c": self.atoms_as_symbols.count("C"),
            "nr_o": self.atoms_as_symbols.count("O"),
            "nr_n": self.atoms_as_symbols.count("N"),
            "nr_f": self.atoms_as_symbols.count("F"),
            "nr_h": self.atoms_as_symbols.count("H"),
            "nr_ring_3": self.ring_sizes.count(3),
            "nr_ring_4": self.ring_sizes.count(4),
            "nr_ring_5": self.ring_sizes.count(5),
            "nr_ring_6": self.ring_sizes.count(6),
            "fully_connected": self.fully_connected,
        }

        if self._name:
            metrics.update({"name": self._name})

        return metrics
