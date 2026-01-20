# GPU-FIRST, CPU-LIGHT Lazy Dataset for Dense Chess Graphs
# ------------------------------------------------------
# This file is designed to AVOID CPU BOTTLENECKS.
# - No preprocessing of graphs
# - No pandas
# - No multiprocessing
# - Dense graph structure preserved EXACTLY
# - Graph is built lazily and mostly on GPU

import h5py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import chess
import csv

# -----------------------------
# STATIC PRECOMPUTE (ONCE)
# -----------------------------

def precompute_queen_knight_edges():
    edges = []
    for sq in chess.SQUARES:
        r = chess.square_rank(sq)
        f = chess.square_file(sq)

        # Knight
        for dr, df in [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]:
            rr, ff = r+dr, f+df
            if 0 <= rr < 8 and 0 <= ff < 8:
                edges.append((sq, chess.square(ff, rr)))

        # Queen
        for dr, df in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            rr, ff = r+dr, f+df
            while 0 <= rr < 8 and 0 <= ff < 8:
                edges.append((sq, chess.square(ff, rr)))
                rr += dr
                ff += df

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


# -----------------------------
# LOOKUP TABLES (STATIC)
# -----------------------------

PIECE_TO_ID = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}


def build_piece_lut():
    # one-hot(12) + value
    lut = torch.zeros((13, 13), dtype=torch.float32)
    values = [0,
              0.1,0.325,0.3,0.5,0.9,1.0,
             -0.1,-0.325,-0.3,-0.5,-0.9,-1.0]

    for i in range(1, 13):
        lut[i, i-1] = 1.0
        lut[i, -1] = values[i]
    return lut


def square_coords():
    coords = []
    for sq in chess.SQUARES:
        coords.append([(sq % 8)/7.0, (sq // 8)/7.0])
    return torch.tensor(coords, dtype=torch.float32)


# -----------------------------
# MAIN DATASET
# -----------------------------

class ChessLazyDenseDataset(Dataset):
    def __init__(self, h5_path: str, device: str = 'cuda'):
        self.h5_path = h5_path
        h5 = h5py.File(h5_path, 'r')   # apri qui, chiudi automaticamente
        self.fen  = h5['fen']
        self.eval = h5['eval']
        self.move = h5['move']
        #with h5py.File(self.h5_path, 'r') as h5: 
        #    self.fen=h5['fen']

        self.device = device  # tienilo per comodità, ma NON usarlo in __getitem__

        # Precomputa SOLO le cose fisse su CPU
        self.edge_index = precompute_queen_knight_edges()          # resta su CPU
        self.piece_lut = build_piece_lut()                         # CPU
        self.coords = square_coords()                              # CPU

        # edge lookup (CPU dict, tiny)
        self.edge_to_idx = {
            (int(a), int(b)): i
            for i, (a, b) in enumerate(self.edge_index.t().cpu().tolist())
        }

        self.length=2413784

    def __len__(self):
        return len(self.fen)

    def _fen_to_piece_tensor(self, fen: str):
        board = chess.Board(fen)
        t = torch.zeros(64, dtype=torch.long)
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                t[sq] = PIECE_TO_ID[p.symbol()]
        return t

    def __getitem__(self, idx):
        fen = self.fen[idx].decode('ascii')
        eval = self.eval[idx]
        move = self.move[idx]

        board = chess.Board(fen)


        piece_ids = self._fen_to_piece_tensor(fen)

        x = self.piece_lut[piece_ids]
        x = torch.cat([x, self.coords], dim=1)


        # VALUE TARGET
        y = torch.tensor([eval], dtype=torch.float32)  # CPU

        # POLICY TARGET (DENSE)
        n_edges = self.edge_index.size(1)
        y_policy = torch.zeros(n_edges, dtype=torch.float32)
        move_code = int(move)
        if move_code >= 0:
            src = move_code // 64
            dst = move_code % 64
            ei = self.edge_to_idx.get((src, dst))
            if ei is not None:
                y_policy[ei] = 1.0


        # LEGAL EDGE MASK (DENSE, SAME GRAPH)
        mask = torch.zeros(n_edges, dtype=torch.bool)
        for m in board.legal_moves:
            ei = self.edge_to_idx.get((m.from_square, m.to_square))
            if ei is not None:
                mask[ei] = 1

        global_features = []
        
        # Turno (chi deve muovere)
        turn = 1.0 if board.turn == chess.WHITE else -1.0
        global_features.append(turn)
        
        # Diritti di arrocco
        global_features.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
        global_features.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
        global_features.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
        global_features.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
        
        # En passant
        global_features.append(1.0 if board.ep_square is not None else 0.0)
        
        # Numero mosse
        global_features.append(board.fullmove_number / 100.0)

        global_features = torch.tensor(global_features, dtype=torch.float)

        return Data(
        x=x,
        edge_index=self.edge_index,
        y=y,
        y_policy=y_policy,
        legal_edge_mask=mask,
        global_features=global_features
        )



def create_hdf5_from_csv(csv_path: str, h5_path: str):
    """
    Expected CSV columns:
    FEN, Evaluation, Move


    Move must be UCI (e2e4). If missing, leave empty.
    Evaluation can be centipawns or mate notation.
    """


    # First pass: count rows
    with open(csv_path, 'r') as f:
        n_samples = sum(1 for _ in f) - 1


    print(f"Creating HDF5 with {n_samples} samples")


    with h5py.File(h5_path, 'w') as h5:
        fen_ds = h5.create_dataset('fen', (n_samples,), dtype=h5py.string_dtype('ascii'))
        eval_ds = h5.create_dataset('eval', (n_samples,), dtype='float32')
        move_ds = h5.create_dataset('move', (n_samples,), dtype='int32')


        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                fen_ds[i] = row['FEN']


                # Evaluation handling
                ev = row['Evaluation']
                if '#' in ev:
                    eval_ds[i] = 1.0 if '+' in ev else -1.0
                else:
                    try:
                        eval_ds[i] = float(ev) / 1000.0
                    except:
                        eval_ds[i] = 0.0


                # Move encoding: src*64 + dst
                mv = row.get('Move', '')
                if mv and len(mv) >= 4:
                    try:
                        src = chess.parse_square(mv[:2])
                        dst = chess.parse_square(mv[2:4])
                        move_ds[i] = src * 64 + dst
                    except:
                        move_ds[i] = -1
                else:
                    move_ds[i] = -1


                if i % 100_000 == 0 and i > 0:
                    print(f" processed {i}/{n_samples}")


    print(f"✓ HDF5 created at: {h5_path}")


class DatasetMPNN(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        # edge_index e edge_attr possono stare fuori dai worker (sono piccoli)
        with h5py.File(h5_path, 'r') as h5:
            self.edge_index = torch.tensor(h5["edge_index"][:], dtype=torch.long)
            self.edge_attr  = torch.tensor(h5["edge_attr"][:], dtype=torch.float32)
        self.piece_lut = build_piece_lut()
        self.coords = torch.stack([torch.arange(8).repeat(8)/7.0,
                                   torch.arange(8).unsqueeze(1).repeat(1,8).flatten()/7.0],1)
        # lunghezza del dataset
        with h5py.File(h5_path, 'r') as h5:
            self.len = len(h5["value"])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Apri e chiudi HDF5 localmente, per ogni sample
        with h5py.File(self.h5_path, 'r') as h5:
            piece_ids = torch.from_numpy(h5["piece_ids"][idx]).long()
            legal_mask = torch.from_numpy(h5["legal_mask"][idx])
            policy_idx = int(h5["policy_edge"][idx])
            value = torch.tensor([h5["value"][idx]], dtype=torch.float32)
            global_feat = torch.from_numpy(h5["global_features"][idx])

        # costruisci x dai piece_ids
        x = self.piece_lut[piece_ids]
        x = torch.cat([x, self.coords], dim=1)

        # costruisci y_policy
        y_pol = torch.zeros(len(legal_mask), dtype=torch.float32)
        if policy_idx >= 0:
            y_pol[policy_idx] = 1.0

        return Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            legal_edge_mask=legal_mask,
            y_policy=y_pol,
            y=value,
            global_features=global_feat
        )
