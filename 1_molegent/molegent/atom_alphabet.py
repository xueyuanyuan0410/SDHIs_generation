from enum import Enum, unique, auto

#ATOMIC_NUMBER_MAPPING = {6: "C", 7: "N", 8: "O", 9: "F", 17: "Cl", 16: "S", 35: "Br"} #old
ATOMIC_NUMBER_MAPPING = {6: "C", 7: "N", 8: "O", 9: "F", 17: "Cl", 14:"Si",16: "S", 35: "Br"} #new

@unique
class Atoms(Enum):
    PAD = 0
    BOM = 1
    EOM = 2
    C = auto()
    O = auto()
    N = auto()
    F = auto()

@unique
class ZincAtoms(Enum):
    PAD = 0
    BOM = 1
    EOM = 2
    C = auto()
    O = auto()
    N = auto()
    F = auto()
    S = auto()
    Cl = auto()
    Br = auto()
    Si = auto()

    @staticmethod
    def from_atomic_number(atomic_number):
        # print('--------------☆----------------------')
        # print(atomic_number)
        # print(ZincAtoms[ATOMIC_NUMBER_MAPPING[atomic_number]])
        return ZincAtoms[ATOMIC_NUMBER_MAPPING[atomic_number]]