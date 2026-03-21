"""
dataset_with_precomputed_fields.py
------------------------------------
Fast dataset that reads precomputed influence fields from HDF5
instead of computing BFS on-the-fly.

Requires:
  - your CSV (FEN, Evaluation, Move)
  - fields.h5 produced by precompute_fields.py

Speed: same as baseline (no CPU bottleneck from BFS).
"""

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import chess

from ..data_organization_tensor import generate_all_legal_move_vocab, collate_fn


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChessDatasetPrecomputed(Dataset):
    """
    Reads board positions from CSV and precomputed fields from HDF5.

    Args:
        csv_path   : path to CSV with FEN, Evaluation, Move columns
        h5_path    : path to HDF5 produced by precompute_fields.py
        move_vocab : dict mapping UCI move strings to indices
        indices    : array of row indices to use (for train/val/test split)
        alpha      : only used for documentation — fields already computed
    """

    def __init__(
        self,
        csv_path:   str,
        h5_path:    str,
        move_vocab: dict,
        indices:    np.ndarray,
    ):
        self.csv_path   = csv_path
        self.h5_path    = h5_path
        self.move_vocab = move_vocab
        self.indices    = indices

        # Load CSV columns into memory (strings are small)
        df = pd.read_csv(csv_path)
        self.fens  = df['FEN'].values
        self.moves = df['Move'].values
        self.evals = df['Evaluation'].values

        # Open HDF5 once — h5py handles concurrent reads fine
        self._h5 = h5py.File(h5_path, 'r', swmr=True)
        self.white_ds   = self._h5['white_field']
        self.black_ds   = self._h5['black_field']
        self.control_ds = self._h5['control_field']

        self.piece_to_plane = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
        }

    def __len__(self):
        return len(self.indices)

    def _parse_eval(self, e: str) -> float:
        e = str(e).strip()
        if '#' in e:
            return 1.0 if '+' in e else -1.0
        try:
            return max(-1000, min(1000, int(e))) / 1000.0
        except ValueError:
            return 0.0

    def __getitem__(self, i):
        idx = int(self.indices[i])

        fen    = self.fens[idx]
        move   = self.moves[idx]
        result = self.evals[idx]

        # ---- Board planes (channels 0-12) ----
        board_planes = torch.zeros((16, 8, 8), dtype=torch.float32)

        board_fen = fen.split(' ')[0]
        turn      = fen.split(' ')[1]

        for rank_idx, row in enumerate(board_fen.split('/')):
            file_idx = 0
            for char in row:
                if char.isdigit():
                    file_idx += int(char)
                elif char in self.piece_to_plane:
                    board_planes[self.piece_to_plane[char], rank_idx, file_idx] = 1
                    file_idx += 1

        board_planes[12, :, :] = 1.0 if turn == 'w' else 0.0

        # ---- Precomputed influence fields (channels 13-15) ----
        board_planes[13] = torch.from_numpy(
            self.white_ds[idx].astype(np.float32))
        board_planes[14] = torch.from_numpy(
            self.black_ds[idx].astype(np.float32))
        board_planes[15] = torch.from_numpy(
            self.control_ds[idx].astype(np.float32))

        # ---- Legal move mask ----
        board       = chess.Board(fen)
        legal_moves = [str(m) for m in board.legal_moves]
        mask        = [0] * 1968
        for m in legal_moves:
            idx_m = self.move_vocab.get(m, -1)
            if idx_m >= 0:
                mask[idx_m] = 1

        # ---- Move encoding ----
        move_encoded = self.move_vocab.get(str(move), -1)

        # ---- Evaluation ----
        result_val = self._parse_eval(result)

        return board_planes, move_encoded, mask, result_val

    def close(self):
        self._h5.close()


# ---------------------------------------------------------------------------
# Baseline dataset (identical speed, 13 channels)
# ---------------------------------------------------------------------------

class ChessDatasetBaseline(Dataset):
    """
    Same interface as ChessDatasetPrecomputed but outputs (13, 8, 8).
    Uses the same index array so train/val splits are identical.
    """

    def __init__(self, csv_path: str, move_vocab: dict, indices: np.ndarray):
        self.move_vocab = move_vocab
        self.indices    = indices

        df = pd.read_csv(csv_path)
        self.fens  = df['FEN'].values
        self.moves = df['Move'].values
        self.evals = df['Evaluation'].values

        self.piece_to_plane = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
        }

    def __len__(self):
        return len(self.indices)

    def _parse_eval(self, e: str) -> float:
        e = str(e).strip()
        if '#' in e:
            return 1.0 if '+' in e else -1.0
        try:
            return max(-1000, min(1000, int(e))) / 1000.0
        except ValueError:
            return 0.0

    def __getitem__(self, i):
        idx = int(self.indices[i])

        fen    = self.fens[idx]
        move   = self.moves[idx]
        result = self.evals[idx]

        board_planes = torch.zeros((13, 8, 8), dtype=torch.float32)

        board_fen = fen.split(' ')[0]
        turn      = fen.split(' ')[1]

        for rank_idx, row in enumerate(board_fen.split('/')):
            file_idx = 0
            for char in row:
                if char.isdigit():
                    file_idx += int(char)
                elif char in self.piece_to_plane:
                    board_planes[self.piece_to_plane[char], rank_idx, file_idx] = 1
                    file_idx += 1

        board_planes[12, :, :] = 1.0 if turn == 'w' else 0.0

        board       = chess.Board(fen)
        legal_moves = [str(m) for m in board.legal_moves]
        mask        = [0] * 1968
        for m in legal_moves:
            idx_m = self.move_vocab.get(m, -1)
            if idx_m >= 0:
                mask[idx_m] = 1

        move_encoded = self.move_vocab.get(str(move), -1)
        result_val   = self._parse_eval(result)

        return board_planes, move_encoded, mask, result_val


# ---------------------------------------------------------------------------
# Factory: creates matched train/val splits for both variants
# ---------------------------------------------------------------------------

def create_matched_dataloaders(
    csv_path:   str,
    h5_path:    str,
    n_samples:  int   = 500_000,
    val_frac:   float = 0.15,
    batch_size: int   = 256,
    n_workers:  int   = 4,
    seed:       int   = 42,
):
    """
    Creates train/val dataloaders for BOTH baseline and fields variants
    using EXACTLY the same positions — guarantees a fair comparison.

    Returns:
        train_base, val_base, train_fields, val_fields, move_vocab
    """
    rng      = np.random.default_rng(seed)
    n_total  = sum(1 for _ in open(csv_path)) - 1
    n_use    = min(n_samples, n_total)

    # Sample indices once — shared by both variants
    all_idx  = rng.choice(n_total, size=n_use, replace=False)
    n_val    = int(n_use * val_frac)
    val_idx  = all_idx[:n_val]
    train_idx = all_idx[n_val:]

    move_vocab = generate_all_legal_move_vocab()

    # Baseline datasets
    train_base_ds = ChessDatasetBaseline(csv_path, move_vocab, train_idx)
    val_base_ds   = ChessDatasetBaseline(csv_path, move_vocab, val_idx)

    # Fields datasets
    train_fields_ds = ChessDatasetPrecomputed(csv_path, h5_path, move_vocab, train_idx)
    val_fields_ds   = ChessDatasetPrecomputed(csv_path, h5_path, move_vocab, val_idx)

    g = torch.Generator()
    g.manual_seed(seed)

    def make_loader(ds, shuffle):
        return torch.utils.data.DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            collate_fn  = collate_fn,
            num_workers = n_workers,
            pin_memory  = True,
            persistent_workers = n_workers > 0,
            generator   = g if shuffle else None,
        )

    train_base   = make_loader(train_base_ds,   shuffle=True)
    val_base     = make_loader(val_base_ds,     shuffle=False)
    train_fields = make_loader(train_fields_ds, shuffle=True)
    val_fields   = make_loader(val_fields_ds,   shuffle=False)

    n_train = len(train_idx)
    n_val_  = len(val_idx)
    print(f'Train: {n_train:,}  |  Val: {n_val_:,}')
    print(f'Batches/epoch: {len(train_base):,}')
    print(f'Same positions for both variants: ✓')

    return train_base, val_base, train_fields, val_fields, move_vocab
