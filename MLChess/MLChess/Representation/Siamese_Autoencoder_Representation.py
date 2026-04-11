import pandas as pd
import chess
import chess.engine
import h5py
import numpy as np
from tqdm import tqdm
import random
from torch.utils.data import Dataset
import torch

STOCKFISH_PATH = "/usr/bin/stockfish"
OUTPUT_H5 = "mate_trajectories.h5"

piece_to_plane = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}

def fen_to_tensor(fen: str) -> np.ndarray:
    board_planes = np.zeros((13, 8, 8), dtype=np.float32)
    board_fen, turn = fen.split(' ')[0], fen.split(' ')[1]
    for rank_idx, row in enumerate(board_fen.split('/')):
        file_idx = 0
        for char in row:
            if char.isdigit():
                file_idx += int(char)
            elif char in piece_to_plane:
                board_planes[piece_to_plane[char], rank_idx, file_idx] = 1
                file_idx += 1
    board_planes[12, :, :] = 1 if turn == 'w' else 0
    return board_planes


def parse_mate_depth(eval_str: str):
    """
    Ritorna (depth, winner) dove winner = 'white' o 'black'.
    #+N → bianco matta in N, #-N → nero matta in N.
    """
    eval_str = eval_str.strip()
    if eval_str.startswith('#+'):
        return int(eval_str[2:]), 'white'
    elif eval_str.startswith('#-'):
        return int(eval_str[2:]), 'black'
    elif eval_str.startswith('#'):
        # fallback per '#3' senza segno esplicito
        return int(eval_str[1:]), 'unknown'
    return None, None


def build_trajectory(fen: str, first_move: str, depth: int, engine: chess.engine.SimpleEngine) -> list[np.ndarray] | None:
    board = chess.Board(fen)
    tensors = [fen_to_tensor(board.fen())]

    try:
        board.push_uci(first_move)
        tensors.append(fen_to_tensor(board.fen()))
    except Exception:
        return None

    if board.is_checkmate():
        return tensors

    for step in range(depth):  # fix: range(depth) non range(depth-1)
        if board.is_game_over():
            break
        try:
            remaining = depth - step
            result = engine.play(
                board,
                chess.engine.Limit(mate=remaining, depth=20)
            )
            if result.move is None:
                return None
            board.push(result.move)
            tensors.append(fen_to_tensor(board.fen()))
        except Exception:
            return None

    if not board.is_checkmate():
        return None

    return tensors


def build_and_save_trajectories(CSV_FILE = "over_mate_1_tactic_evals.csv"):
    df = pd.read_csv(CSV_FILE)
    mate_df = df[df["Evaluation"].str.contains("#")].reset_index(drop=True)
    print(f"Posizioni di matto trovate: {len(mate_df)}")

    trajectories = []   # lista di np.ndarray shape (N_i, 13, 8, 8)
    winners = []        # 'white' o 'black' per ogni traiettoria
    lengths = []        # lunghezza di ogni traiettoria

    failed_checkmate = 0
    failed_move = 0
    failed_depth = 0

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        for _, row in tqdm(mate_df.iterrows(), total=len(mate_df)):
            depth, winner = parse_mate_depth(row["Evaluation"])
            
            if depth is None or depth > 6:
                failed_depth += 1
                continue

            traj = build_trajectory(row["FEN"], row["Move"], depth, engine)
            
            if traj is None:
                failed_checkmate += 1
                continue

            trajectories.append(np.stack(traj))
            winners.append(winner)
            lengths.append(len(traj))

    print(f"Saltate per depth > 6: {failed_depth}")
    print(f"Fallite per matto non raggiunto: {failed_checkmate}")
    print(f"Costruite: {len(trajectories)}")

    print(f"Traiettorie costruite: {len(trajectories)}")
    print(f"Lunghezze: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

    # Salvataggio in HDF5 — ogni traiettoria è un dataset separato
    # perché hanno lunghezze diverse
    with h5py.File(OUTPUT_H5, 'w') as f:
        for i, (traj, winner, length) in enumerate(zip(trajectories, winners, lengths)):
            ds = f.create_dataset(f"traj_{i}", data=traj, compression="gzip")
            ds.attrs["winner"] = winner
            ds.attrs["length"] = length

        f.attrs["n_trajectories"] = len(trajectories)
        print(f"Salvato in {OUTPUT_H5}")


class SiameseChessDataset(Dataset):
    """
    Costruisce coppie (board_A, board_B, label) dove:
    - label = 1 → stessa traiettoria (positivo)
    - label = 0 → traiettorie diverse (negativo)
    """
    def __init__(self, h5_path: str, n_pairs: int = 300_000, seed: int = 42):
        self.h5_path = h5_path
        self.n_pairs = n_pairs
        random.seed(seed)
        np.random.seed(seed)

        with h5py.File(h5_path, 'r') as f:
            self.n_traj = f.attrs["n_trajectories"]
            self.lengths = [f[f"traj_{i}"].attrs["length"] for i in range(self.n_traj)]
            self.winners = [f[f"traj_{i}"].attrs["winner"] for i in range(self.n_traj)]

        self.valid_trajs = [i for i, l in enumerate(self.lengths) if l >= 2]
        self.pairs = self._generate_pairs()

    def _generate_pairs(self):
        pairs = []
        n_pos = self.n_pairs // 2
        n_neg = self.n_pairs - n_pos

        # Coppie positive: due posizioni dalla stessa traiettoria
        for _ in range(n_pos):
            t = random.choice(self.valid_trajs)
            i, j = random.sample(range(self.lengths[t]), 2)
            pairs.append((t, i, t, j, 1.0))

        # Coppie negative: traiettorie diverse
        for _ in range(n_neg):
            t1, t2 = random.sample(range(self.n_traj), 2)
            i = random.randint(0, self.lengths[t1] - 1)
            j = random.randint(0, self.lengths[t2] - 1)
            pairs.append((t1, i, t2, j, 0.0))

        random.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        t_a, i_a, t_b, i_b, label = self.pairs[idx]

        with h5py.File(self.h5_path, 'r') as f:
            board_a = torch.tensor(f[f"traj_{t_a}"][i_a], dtype=torch.float32)
            board_b = torch.tensor(f[f"traj_{t_b}"][i_b], dtype=torch.float32)

        return board_a, board_b, torch.tensor(label, dtype=torch.float32)