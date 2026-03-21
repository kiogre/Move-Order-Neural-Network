"""
data_organization_tensor_fields.py
------------------------------------
Drop-in replacement for data_organization_tensor.py.
Adds 3 extra channels (white_field, black_field, control_field) to the
board tensor: shape goes from (13, 8, 8) to (16, 8, 8).

Usage is identical to the original — just import from this file instead.
"""

import chess
import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

from .chess_fields import compute_fields_from_fen   # <- your module


# ---------------------------------------------------------------------------
# Move vocab (unchanged from original)
# ---------------------------------------------------------------------------

def generate_all_legal_move_vocab() -> dict[str, int]:
    move_dict = {}
    index = 0
    promotion_pieces = ['q', 'r', 'b', 'n']

    for color in [chess.WHITE, chess.BLACK]:
        for from_square in chess.SQUARES:
            from_name = chess.square_name(from_square)
            file_from = chess.square_file(from_square)
            rank_from = chess.square_rank(from_square)

            for piece_type in [chess.PAWN, chess.KNIGHT, chess.QUEEN]:

                if piece_type == chess.KNIGHT:
                    deltas = [(-1,-2),(1,-2),(-2,-1),(2,-1),(-2,1),(2,1),(-1,2),(1,2)]
                    for df, dr in deltas:
                        f = file_from + df
                        r = rank_from + dr
                        if 0 <= f < 8 and 0 <= r < 8:
                            move_str = from_name + chess.square_name(chess.square(f, r))
                            if move_str not in move_dict:
                                move_dict[move_str] = index; index += 1

                elif piece_type == chess.QUEEN:
                    for df, dr in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                        f, r = file_from, rank_from
                        while True:
                            f += df; r += dr
                            if not (0 <= f < 8 and 0 <= r < 8): break
                            move_str = from_name + chess.square_name(chess.square(f, r))
                            if move_str not in move_dict:
                                move_dict[move_str] = index; index += 1

                elif piece_type == chess.PAWN:
                    if color == chess.WHITE:
                        if rank_from < 7:
                            to_name = chess.square_name(chess.square(file_from, rank_from+1))
                            move_str = from_name + to_name
                            if move_str not in move_dict:
                                move_dict[move_str] = index; index += 1
                            if rank_from == 1:
                                to_name = chess.square_name(chess.square(file_from, 3))
                                move_str = from_name + to_name
                                if move_str not in move_dict:
                                    move_dict[move_str] = index; index += 1
                            if rank_from == 6:
                                for promo in promotion_pieces:
                                    move_str = from_name + chess.square_name(chess.square(file_from, 7)) + promo
                                    if move_str not in move_dict:
                                        move_dict[move_str] = index; index += 1
                            for df in [-1, 1]:
                                if 0 <= file_from + df < 8:
                                    to_name = chess.square_name(chess.square(file_from+df, rank_from+1))
                                    move_str = from_name + to_name
                                    if move_str not in move_dict:
                                        move_dict[move_str] = index; index += 1
                                    if rank_from == 6:
                                        for promo in promotion_pieces:
                                            move_str = from_name + to_name + promo
                                            if move_str not in move_dict:
                                                move_dict[move_str] = index; index += 1
                    else:
                        if rank_from > 0:
                            to_name = chess.square_name(chess.square(file_from, rank_from-1))
                            move_str = from_name + to_name
                            if move_str not in move_dict:
                                move_dict[move_str] = index; index += 1
                            if rank_from == 6:
                                to_name = chess.square_name(chess.square(file_from, 4))
                                move_str = from_name + to_name
                                if move_str not in move_dict:
                                    move_dict[move_str] = index; index += 1
                            if rank_from == 1:
                                for promo in promotion_pieces:
                                    move_str = from_name + chess.square_name(chess.square(file_from, 0)) + promo
                                    if move_str not in move_dict:
                                        move_dict[move_str] = index; index += 1
                            for df in [-1, 1]:
                                if 0 <= file_from + df < 8:
                                    to_name = chess.square_name(chess.square(file_from+df, rank_from-1))
                                    move_str = from_name + to_name
                                    if move_str not in move_dict:
                                        move_dict[move_str] = index; index += 1
                                    if rank_from == 1:
                                        for promo in promotion_pieces:
                                            move_str = from_name + to_name + promo
                                            if move_str not in move_dict:
                                                move_dict[move_str] = index; index += 1
    return move_dict


# ---------------------------------------------------------------------------
# Transform: (13, 8, 8) board  +  (3, 8, 8) influence fields = (16, 8, 8)
# ---------------------------------------------------------------------------

class ChessTransformWithFields:
    """
    Same interface as ChessTransform but outputs (16, 8, 8) tensors.

    Channels:
      0-11  : one-hot piece planes (P N B R Q K p n b r q k)
      12    : turn plane (1=white, 0=black)
      13    : white influence field  Φ_w(x)
      14    : black influence field  Φ_b(x)
      15    : control field          C(x) = Φ_w - Φ_b
    """

    def __init__(self, move_vocab: dict, alpha: float = 0.5):
        self.move_vocab = move_vocab
        self.alpha = alpha
        self.piece_to_plane = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
        }

    def __call__(self, position: str, move: str, legal_indices: list, result: str):
        # ---- board planes (channels 0-12) ----
        board_planes = torch.zeros((16, 8, 8), dtype=torch.float32)

        board_fen = position.split(' ')[0]
        turn      = position.split(' ')[1]

        for rank_idx, row in enumerate(board_fen.split('/')):
            file_idx = 0
            for char in row:
                if char.isdigit():
                    file_idx += int(char)
                elif char in self.piece_to_plane:
                    board_planes[self.piece_to_plane[char], rank_idx, file_idx] = 1
                    file_idx += 1

        board_planes[12, :, :] = 1.0 if turn == 'w' else 0.0

        # ---- influence fields (channels 13-15) ----
        try:
            fields = compute_fields_from_fen(position, alpha=self.alpha)
            board_planes[13] = torch.from_numpy(fields['white'].astype(np.float32))
            board_planes[14] = torch.from_numpy(fields['black'].astype(np.float32))
            board_planes[15] = torch.from_numpy(fields['control'].astype(np.float32))
        except Exception:
            pass  # leave as zeros if parsing fails

        # ---- legal move mask ----
        mask = [0] * 1968
        for idx in legal_indices:
            if 0 <= idx < 1968:
                mask[idx] = 1

        # ---- move encoding ----
        move_encoded = self.move_vocab.get(move, -1)

        # ---- evaluation ----
        max_cp = 1000
        r = str(result).strip()
        if '#' in r:
            result_val = 1.0 if '+' in r else -1.0
        else:
            try:
                result_val = max(-max_cp, min(max_cp, int(r))) / max_cp
            except ValueError:
                result_val = 0.0

        return board_planes, move_encoded, mask, result_val


# ---------------------------------------------------------------------------
# Dataset (identical logic to original ChessDataset)
# ---------------------------------------------------------------------------

class ChessDatasetWithFields(torch.utils.data.Dataset):
    def __init__(self, csv_file: str, move_vocab: dict, split: str = 'train',
                 transform=None):
        self.df = pd.read_csv(csv_file)

        total_len = len(self.df)
        train_end = int(total_len * 0.7)
        val_end   = int(total_len * 0.85)

        if split == 'train':
            self.df = self.df.iloc[:train_end]
        elif split == 'validation':
            self.df = self.df.iloc[train_end:val_end]
        elif split == 'test':
            self.df = self.df.iloc[val_end:]
        else:
            raise ValueError("split must be 'train', 'validation', or 'test'")

        self.df        = self.df.reset_index(drop=True)
        self.move_vocab = move_vocab
        self.transform  = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row      = self.df.iloc[index]
        position = row['FEN']
        move     = row['Move']
        result   = row['Evaluation']

        board       = chess.Board(position)
        legal_moves = [str(m) for m in board.legal_moves]
        mask        = [self.move_vocab.get(m, -1) for m in legal_moves]

        if self.transform is not None:
            position, move, mask, result = self.transform(position, move, mask, result)

        return position, move, mask, result


# ---------------------------------------------------------------------------
# collate_fn (identical to original)
# ---------------------------------------------------------------------------

def collate_fn(batch):
    positions, moves, mask, result = zip(*batch)
    positions        = [torch.as_tensor(p) for p in positions]
    positions_padded = pad_sequence(positions, batch_first=True, padding_value=0)
    moves            = torch.tensor(moves,  dtype=torch.long)
    mask             = torch.tensor(mask,   dtype=torch.bool)
    result           = torch.tensor(result, dtype=torch.float)
    return positions_padded, moves, mask, result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_dataloaders_with_fields(
    name_file:  str   = 'your_dataset.csv',
    batch_size: int   = 128,
    alpha:      float = 0.5,
):
    """
    Drop-in replacement for create_dataloaders_tensor().
    Returns (trainloader, validationloader, testloader, move_vocab).
    Board tensors are now (16, 8, 8) instead of (13, 8, 8).
    """
    all_moves  = generate_all_legal_move_vocab()
    move_vocab = {move: idx for idx, move in enumerate(all_moves)}

    transform = ChessTransformWithFields(move_vocab=move_vocab, alpha=alpha)

    trainset      = ChessDatasetWithFields(name_file, move_vocab, 'train',      transform)
    validationset = ChessDatasetWithFields(name_file, move_vocab, 'validation', transform)
    testset       = ChessDatasetWithFields(name_file, move_vocab, 'test',       transform)

    g = torch.Generator()

    trainloader = torch.utils.data.DataLoader(
        trainset,      batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, generator=g)
    validationloader = torch.utils.data.DataLoader(
        validationset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, generator=g)
    testloader = torch.utils.data.DataLoader(
        testset,       batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, generator=g)

    print(f"Train set size:      {len(trainset)}")
    print(f"Validation set size: {len(validationset)}")
    print(f"Test set size:       {len(testset)}")
    print(f"Board tensor shape:  (16, 8, 8)  [13 original + 3 field channels]")

    return trainloader, validationloader, testloader, move_vocab
