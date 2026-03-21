"""
new_graph_representation_fields.py
------------------------------------
Drop-in replacement for new_graph_representation.py.

Changes vs original:
  - Node feature vector grows from 15 → 18 dims:
      [one_hot(12), piece_value, coord_x, coord_y,
       white_field, black_field, control_field]   ← 3 new dims
  - ChessLazyDenseDataset reads precomputed fields from a second HDF5
    (fields.h5 produced by precompute_fields.py), so no BFS at getitem time.
  - Fallback: if no fields HDF5 is provided, fields are computed on-the-fly.
  - Everything else (edge_index, legal_mask, y_policy, y, global_features)
    is identical to the original.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import chess

from .chess_fields import compute_fields_from_fen   # fallback only


# ---------------------------------------------------------------------------
# Static precomputes (identical to original)
# ---------------------------------------------------------------------------

def precompute_queen_knight_edges():
    edges = []
    for sq in chess.SQUARES:
        r = chess.square_rank(sq)
        f = chess.square_file(sq)
        for dr, df in [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]:
            rr, ff = r+dr, f+df
            if 0 <= rr < 8 and 0 <= ff < 8:
                edges.append((sq, chess.square(ff, rr)))
        for dr, df in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            rr, ff = r+dr, f+df
            while 0 <= rr < 8 and 0 <= ff < 8:
                edges.append((sq, chess.square(ff, rr)))
                rr += dr; ff += df
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


PIECE_TO_ID = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}


def build_piece_lut():
    """one_hot(12) + piece_value → shape (13, 13)"""
    lut = torch.zeros((13, 13), dtype=torch.float32)
    values = [0, 0.1, 0.325, 0.3, 0.5, 0.9, 1.0,
                -0.1,-0.325,-0.3,-0.5,-0.9,-1.0]
    for i in range(1, 13):
        lut[i, i-1] = 1.0
        lut[i, -1]  = values[i]
    return lut


def square_coords():
    coords = []
    for sq in chess.SQUARES:
        coords.append([(sq % 8) / 7.0, (sq // 8) / 7.0])
    return torch.tensor(coords, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class ChessLazyDenseDatasetWithFields(Dataset):
    """
    Same as ChessLazyDenseDataset but with influence fields as extra node features.

    Node feature vector (per square):
      dims  0-12 : one_hot(piece) + piece_value   (from piece_lut)
      dims 13-14 : square coordinates (x/7, y/7)
      dim  15    : white influence field  Φ_w(x)
      dim  16    : black influence field  Φ_b(x)
      dim  17    : control field          C(x)

    Args:
        h5_path    : path to your existing positions HDF5 (fen, eval, move)
        fields_h5  : path to precomputed fields HDF5 (from precompute_fields.py)
                     If None, fields are computed on-the-fly (slower).
        alpha      : decay parameter (used only for on-the-fly fallback)
    """

    def __init__(self, h5_path: str, fields_h5: str = None, alpha: float = 0.5):
        self.h5_path   = h5_path
        self.fields_h5 = fields_h5
        self.alpha     = alpha

        h5 = h5py.File(h5_path, 'r')
        self.fen  = h5['fen']
        self.eval = h5['eval']
        self.move = h5['move']

        if fields_h5 is not None:
            fh5 = h5py.File(fields_h5, 'r')
            self.white_f   = fh5['white_field']
            self.black_f   = fh5['black_field']
            self.control_f = fh5['control_field']
            self._use_precomputed = True
            print(f"Using precomputed fields from {fields_h5}")
        else:
            self._use_precomputed = False
            print("No fields HDF5 provided — computing on-the-fly (slow).")

        self.edge_index  = precompute_queen_knight_edges()
        self.piece_lut   = build_piece_lut()
        self.coords      = square_coords()
        self.edge_to_idx = {
            (int(a), int(b)): i
            for i, (a, b) in enumerate(self.edge_index.t().cpu().tolist())
        }

    def __len__(self):
        return len(self.fen)

    def _get_fields(self, idx: int, fen: str):
        """Returns (white, black, control) as (64,) tensors."""
        if self._use_precomputed:
            w = torch.from_numpy(self.white_f  [idx].flatten())
            b = torch.from_numpy(self.black_f  [idx].flatten())
            c = torch.from_numpy(self.control_f[idx].flatten())
        else:
            try:
                fields = compute_fields_from_fen(fen, alpha=self.alpha)
                # chess_fields uses row0=rank8; square index in python-chess
                # uses sq=0 → a1 (rank0,file0). We need to reorder.
                # python-chess: sq = rank*8 + file  (rank 0 = rank1)
                # chess_fields board: row r → rank (7-r)
                # So board[r,c] corresponds to sq = (7-r)*8 + c
                def reorder(arr):
                    out = np.zeros(64, dtype=np.float32)
                    for sq in range(64):
                        rank = sq // 8
                        file = sq % 8
                        row  = 7 - rank
                        out[sq] = arr[row, file]
                    return out
                w = torch.from_numpy(reorder(fields['white']))
                b = torch.from_numpy(reorder(fields['black']))
                c = torch.from_numpy(reorder(fields['control']))
            except Exception:
                w = b = c = torch.zeros(64)
        return w, b, c

    def __getitem__(self, idx):
        fen      = self.fen [idx].decode('ascii')
        eval_val = self.eval[idx]
        move_val = self.move[idx]

        board    = chess.Board(fen)

        # Node features: piece encoding
        piece_ids = torch.zeros(64, dtype=torch.long)
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                piece_ids[sq] = PIECE_TO_ID[p.symbol()]

        x_base = self.piece_lut[piece_ids]          # (64, 13)
        x_base = torch.cat([x_base, self.coords], dim=1)  # (64, 15)

        # Influence field features
        w_field, b_field, c_field = self._get_fields(idx, fen)
        x_fields = torch.stack([w_field, b_field, c_field], dim=1)  # (64, 3)

        x = torch.cat([x_base, x_fields], dim=1)   # (64, 18)

        # Value target
        y = torch.tensor([eval_val], dtype=torch.float32)

        # Policy target
        n_edges  = self.edge_index.size(1)
        y_policy = torch.zeros(n_edges, dtype=torch.float32)
        move_code = int(move_val)
        if move_code >= 0:
            src = move_code // 64
            dst = move_code % 64
            ei  = self.edge_to_idx.get((src, dst))
            if ei is not None:
                y_policy[ei] = 1.0

        # Legal edge mask
        mask = torch.zeros(n_edges, dtype=torch.bool)
        for m in board.legal_moves:
            ei = self.edge_to_idx.get((m.from_square, m.to_square))
            if ei is not None:
                mask[ei] = True

        # Global features (identical to original)
        gf = [
            1.0 if board.turn == chess.WHITE else -1.0,
            1.0 if board.has_kingside_castling_rights(chess.WHITE)  else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.BLACK)  else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
            1.0 if board.ep_square is not None else 0.0,
            board.fullmove_number / 100.0,
            # New: aggregate field stats as global scalars
            float(w_field.sum()),    # total white influence
            float(b_field.sum()),    # total black influence
            float(c_field.sum()),    # net control (positive = white)
            float(c_field.abs().max()),  # peak tension
        ]
        global_features = torch.tensor(gf, dtype=torch.float32)

        return Data(
            x                = x,
            edge_index       = self.edge_index,
            y                = y,
            y_policy         = y_policy,
            legal_edge_mask  = mask,
            global_features  = global_features,
        )
