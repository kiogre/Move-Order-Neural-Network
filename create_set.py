import h5py
import chess
import torch
import numpy as np
from tqdm import tqdm

# ----------------------------
# Parametri
# ----------------------------
CSV_PATH = "over_mate_1_tactic_evals.csv"       # CSV originale: FEN, Evaluation, Move
H5_PATH  = "chess_precomputed.h5"
N_EDGES  = 64*64  # massimo teorico (poi useremo solo queen+knight)
N_SAMPLES_ESTIMATE = 2413784

# ----------------------------
# Costruzione grafo e edge_attr
# ----------------------------
edges = []
edge_attr = []

for sq in range(64):
    r, f = divmod(sq, 8)
    # knight jumps
    for dr, df in [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]:
        rr, ff = r+dr, f+df
        if 0 <= rr < 8 and 0 <= ff < 8:
            dst = 8*rr + ff
            edges.append((sq, dst))
            edge_attr.append([dr/7.0, df/7.0, 0, 1])  # knight

    # queen lines
    for dr, df in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
        rr, ff = r+dr, f+df
        while 0 <= rr < 8 and 0 <= ff < 8:
            dst = 8*rr + ff
            edges.append((sq, dst))
            edge_attr.append([dr/7.0, df/7.0, 1, 0])  # queen
            rr += dr
            ff += df

edge_index = np.array(edges, dtype=np.int32).T
edge_attr  = np.array(edge_attr, dtype=np.float32)
N_EDGES = edge_index.shape[1]
print("Grafo creato:", edge_index.shape, edge_attr.shape)

# ----------------------------
# Piece LUT
# ----------------------------
PIECE_TO_ID = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}

def fen_to_piece_ids(fen):
    board = chess.Board(fen)
    t = np.zeros(64, dtype=np.int8)
    for sq in range(64):
        p = board.piece_at(sq)
        if p:
            t[sq] = PIECE_TO_ID[p.symbol()]
    return t

def global_features(board):
    gf = []
    # turno
    gf.append(1.0 if board.turn==chess.WHITE else -1.0)
    # castling
    gf.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    gf.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    gf.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    gf.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
    # en passant
    gf.append(1.0 if board.ep_square is not None else 0.0)
    # numero mosse
    gf.append(board.fullmove_number / 100.0)
    return np.array(gf, dtype=np.float32)

# ----------------------------
# Creazione HDF5
# ----------------------------
with h5py.File(H5_PATH, "w") as h5:
    # datasets
    piece_ds = h5.create_dataset("piece_ids", (N_SAMPLES_ESTIMATE,64), dtype="int8")
    legal_mask_ds = h5.create_dataset("legal_mask", (N_SAMPLES_ESTIMATE,N_EDGES), dtype="bool")
    policy_ds = h5.create_dataset("policy_edge", (N_SAMPLES_ESTIMATE,), dtype="int32")
    value_ds = h5.create_dataset("value", (N_SAMPLES_ESTIMATE,), dtype="float32")
    global_ds = h5.create_dataset("global_features", (N_SAMPLES_ESTIMATE,7), dtype="float32")
    edge_index_ds = h5.create_dataset("edge_index", data=edge_index, dtype="int32")
    edge_attr_ds = h5.create_dataset("edge_attr", data=edge_attr, dtype="float32")

    # edge lookup dict (temporaneo)
    edge_to_idx = { (int(a),int(b)):i for i,(a,b) in enumerate(edges) }

    import csv
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for i,row in enumerate(tqdm(reader, total=N_SAMPLES_ESTIMATE)):
            fen = row["FEN"]
            ev  = row["Evaluation"]
            move_str = row.get("Move","")

            board = chess.Board(fen)
            piece_ds[i] = fen_to_piece_ids(fen)
            global_ds[i] = global_features(board)

            # legal mask
            mask = np.zeros(N_EDGES, dtype=bool)
            for m in board.legal_moves:
                ei = edge_to_idx.get((m.from_square, m.to_square))
                if ei is not None:
                    mask[ei] = True
            legal_mask_ds[i] = mask

            # policy edge
            if move_str and len(move_str)>=4:
                src = chess.parse_square(move_str[:2])
                dst = chess.parse_square(move_str[2:4])
                ei = edge_to_idx.get((src,dst), -1)
                policy_ds[i] = ei
            else:
                policy_ds[i] = -1

            # value
            if "#" in ev:
                value_ds[i] = 1.0 if "+" in ev else -1.0
            else:
                try:
                    value_ds[i] = float(ev)/1000.0
                except:
                    value_ds[i] = 0.0
