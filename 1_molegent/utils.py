import logging
import math
import time

import torch
from torch import nn
from torch.nn import functional as F


class ModelSaver:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, save_dir: str, patience=20, delta=0.001):
        """
        Saves the model every time a the loss improves. Also features early stopping.
        :param model: The neural network model
        :param save_dir: Path where the model will be saved
        :param patience: Max number an consecutive steps without improvement
        :param delta: Minimum definition of improvement
        """
        self._patience_counter = 0

        self._save_dir = save_dir
        self._delta = delta
        self._model = model
        self._optimizer = optimizer

        self._best_loss = None
        self._patience = patience

    def step(self, cur_loss):

        if self._best_loss is None or cur_loss < self._best_loss - self._delta:
            self._best_loss = cur_loss
            self._patience_counter = 0

            # save the checkpoint
            path = f"{self._save_dir}/checkpoint.pt"
            opti_path = f"{self._save_dir}/optimizer.pt"
            torch.save(obj=self._model.state_dict(), f=path)
            torch.save(obj=self._optimizer.state_dict(), f=opti_path)

            logging.getLogger("main").info(f"saving checkpoint due to new min test loss: {cur_loss}")

        else:
            self._patience_counter += 1
            print(f"Test loss did not decrease. Patience: {self._patience_counter} / {self._patience}")

        return self._patience_counter == self._patience


class PositionalEncoding(nn.Module):

    def __init__(self, max_nr_atoms, d_model, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        max_len = max_nr_atoms ** 2
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # pe is of shape [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor[N, S, E]

        Shapes:
            N: batch size
            S: src sequence length
            E: embedding size
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PadCollate:
    def __init__(self, pad_token_atom: int, pad_value_position: int):
        self.pad_token = pad_token_atom
        self.pad_value_position = pad_value_position

    def __call__(self, input_target_tuples) -> dict:
        """
        Collate function for a list of token index lists with different lengths.
        """

        src_atom_seqs = [t["src_atoms"] for t in input_target_tuples]
        tgt_atom_seqs = [t["tgt_atoms"] for t in input_target_tuples]
        src_conn_seq = [t["src_connections"] for t in input_target_tuples]
        tgt_conn_seq = [t["tgt_connections"] for t in input_target_tuples]

        lens = [len(x) for x in src_atom_seqs]
        max_len = max(lens)
        pad = [max_len - x for x in lens]

        src_atom_seqs = [F.pad(x, (0, n)) for x, n in zip(src_atom_seqs, pad)]
        tgt_atom_seqs = [F.pad(x, (0, n)) for x, n in zip(tgt_atom_seqs, pad)]

        if src_conn_seq[0].ndim == 1:
            src_conn_seq = [F.pad(x, (0, n)) for x, n in zip(src_conn_seq, pad)]
            tgt_conn_seq = [F.pad(x, (0, n)) for x, n in zip(tgt_conn_seq, pad)]
        else:

            src_conn_seq = [F.pad(x, (0, 0, 0, n)) for x, n in zip(src_conn_seq, pad)]
            tgt_conn_seq = [F.pad(x, (0, 0, 0, n)) for x, n in zip(tgt_conn_seq, pad)]

        tgt_atom_seqs = torch.stack(tgt_atom_seqs)

        return {
            "x": {
                "atoms": torch.stack(src_atom_seqs),
                "connections": torch.stack(src_conn_seq),
                "tgt_atoms": tgt_atom_seqs,
            },
            "tgt_atoms": tgt_atom_seqs,
            "tgt_connections": torch.stack(tgt_conn_seq),
        }



def create_source_mask(size, device):
    """
    create an autoregressive additive mask e.g. for size 3

     [ [0, -inf, -inf],
       [0,    0, -inf],
       [0,    0,    0],
     ]
    """
    mask = torch.triu(torch.ones((size, size), device=device), 1)
    mask[mask == 1] = float("-inf")
    return mask


def adjacency_to_edge_list(amat):

    edge_list = []

    for i in range(amat.shape[0]):
        for j in range(amat.shape[1]):
            if amat[i, j]:
                edge_list.append((i, j))

    return edge_list


class TransposeNBatchNorm(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_feature)

    def forward(self, x):
        x = x.permute((0, 2, 1))
        x = self.bn(x)
        x = x.permute((0, 2, 1))

        return x

