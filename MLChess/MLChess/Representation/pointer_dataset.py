import chess
import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

PIECE_TO_PLANE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}

# Indici per i pezzi nel move vector (one-hot, 6 valori)
PIECE_TYPE_TO_IDX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

# Indici per la promozione nel move vector (one-hot, 5 valori)
# 0 = nessuna promozione, 1 = donna, 2 = torre, 3 = alfiere, 4 = cavallo
PROMO_TO_IDX = {
    None: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}

MOVE_VECTOR_DIM = 46  # Dimensione del vettore di encoding di ogni mossa
# Breakdown:
#   piece_type:  6  (one-hot)
#   from_row:    8  (one-hot)
#   from_col:    8  (one-hot)
#   to_row:      8  (one-hot)
#   to_col:      8  (one-hot)
#   capture:     1  (0/1)
#   en_passant:  1  (0/1)
#   castling:    1  (0/1)
#   promotion:   5  (one-hot, include "no promotion")
# Totale:       46


# ---------------------------------------------------------------------------
# Funzioni di encoding
# ---------------------------------------------------------------------------

def encode_board(fen: str) -> torch.Tensor:
    """
    Codifica una posizione FEN in un tensore (13, 8, 8).

    Piani 0-11: pezzi (6 bianchi + 6 neri)
    Piano 12:   turno (1.0 = bianco, 0.0 = nero)
    """
    board_planes = torch.zeros((13, 8, 8), dtype=torch.float32)

    board_fen, turn = fen.split(' ')[0], fen.split(' ')[1]

    rows = board_fen.split('/')
    for rank_idx, row in enumerate(rows):
        file_idx = 0
        for char in row:
            if char.isdigit():
                file_idx += int(char)
            elif char in PIECE_TO_PLANE:
                board_planes[PIECE_TO_PLANE[char], rank_idx, file_idx] = 1.0
                file_idx += 1

    board_planes[12, :, :] = 1.0 if turn == 'w' else 0.0

    return board_planes


def encode_move(move: chess.Move, board: chess.Board) -> torch.Tensor:
    """
    Codifica una singola mossa come vettore float32 di dimensione MOVE_VECTOR_DIM (46).

    Layout:
        [0:6]   piece_type  (one-hot, 6 valori)
        [6:14]  from_row    (one-hot, 8 valori)
        [14:22] from_col    (one-hot, 8 valori)
        [22:30] to_row      (one-hot, 8 valori)
        [30:38] to_col      (one-hot, 8 valori)
        [38]    capture     (0/1)
        [39]    en_passant  (0/1)
        [40]    castling    (0/1)
        [41:46] promotion   (one-hot, 5 valori: none/Q/R/B/N)
    """
    vec = torch.zeros(MOVE_VECTOR_DIM, dtype=torch.float32)

    from_sq = move.from_square
    to_sq   = move.to_square

    from_row = chess.square_rank(from_sq)
    from_col = chess.square_file(from_sq)
    to_row   = chess.square_rank(to_sq)
    to_col   = chess.square_file(to_sq)

    # Tipo di pezzo sulla casa di partenza
    piece = board.piece_at(from_sq)
    if piece is not None:
        vec[PIECE_TYPE_TO_IDX[piece.piece_type]] = 1.0

    # Casa di partenza
    vec[6  + from_row] = 1.0
    vec[14 + from_col] = 1.0

    # Casa di arrivo
    vec[22 + to_row] = 1.0
    vec[30 + to_col] = 1.0

    # Cattura
    vec[38] = 1.0 if board.is_capture(move) else 0.0

    # En passant
    vec[39] = 1.0 if board.is_en_passant(move) else 0.0

    # Arrocco
    vec[40] = 1.0 if board.is_castling(move) else 0.0

    # Promozione (one-hot a 5 valori)
    promo_idx = PROMO_TO_IDX.get(move.promotion, 0)
    vec[41 + promo_idx] = 1.0

    return vec


def encode_legal_moves(board: chess.Board) -> torch.Tensor:
    """
    Restituisce un tensore (N, 46) con l'encoding di tutte le N mosse legali
    nella posizione corrente, nello stesso ordine di board.legal_moves.
    """
    moves = list(board.legal_moves)
    if not moves:
        return torch.zeros((0, MOVE_VECTOR_DIM), dtype=torch.float32)

    encoded = [encode_move(m, board) for m in moves]
    return torch.stack(encoded, dim=0)  # (N, 46)


def encode_result(result: str, max_cp: int = 1000) -> float:
    """
    Normalizza la valutazione della posizione in [-1, 1].
    Matto positivo → 1.0, matto negativo → -1.0.
    Centipawn clampati a ±max_cp poi divisi per max_cp.
    """
    if '#' in result:
        return 1.0 if '+' in result else -1.0
    return max(-max_cp, min(max_cp, int(result))) / max_cp


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PointerChessDataset(torch.utils.data.Dataset):
    """
    Dataset per pointer network su scacchi.

    Ogni elemento restituisce:
        board_tensor  : (13, 8, 8)          encoding della posizione
        legal_moves   : (N, 46)             encoding delle N mosse legali
        target_idx    : int                 indice della mossa target in legal_moves
        result        : float               valutazione normalizzata in [-1, 1]
    """

    def __init__(self, csv_file: str, split: str = 'train'):
        """
        Args:
            csv_file : percorso al CSV con colonne FEN, Move, Evaluation
            split    : 'train', 'validation', o 'test'
        """
        df = pd.read_csv(csv_file)

        total_len = len(df)
        train_end = int(total_len * 0.70)
        val_end   = int(total_len * 0.85)

        if split == 'train':
            df = df.iloc[:train_end]
        elif split == 'validation':
            df = df.iloc[train_end:val_end]
        elif split == 'test':
            df = df.iloc[val_end:]
        else:
            raise ValueError("split deve essere 'train', 'validation', o 'test'")

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row        = self.df.iloc[index]
        fen        = row["FEN"]
        move_uci   = row["Move"]
        result_str = row["Evaluation"]

        board = chess.Board(fen)

        # Encoding della board
        board_tensor = encode_board(fen)

        # Lista mosse legali e loro encoding
        legal_moves_list = list(board.legal_moves)
        legal_moves_tensor = encode_legal_moves(board)  # (N, 46)

        # Indice della mossa target nell'elenco delle mosse legali
        target_move = chess.Move.from_uci(move_uci)
        try:
            target_idx = legal_moves_list.index(target_move)
        except ValueError:
            # Fallback: se la mossa non è trovata (non dovrebbe succedere con dati puliti),
            # usa la prima mossa legale disponibile
            target_idx = 0

        result = encode_result(result_str)

        return board_tensor, legal_moves_tensor, target_idx, result


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Combina un batch di elementi con numero variabile di mosse legali.

    Applica padding alle sequenze di mosse e costruisce una attention mask
    booleana per ignorare le posizioni di padding durante l'attenzione.

    Returns:
        board_tensors  : (B, 13, 8, 8)
        legal_moves    : (B, N_max, 46)   — paddato con zeri
        attention_mask : (B, N_max)        — True = reale, False = padding
        target_indices : (B,)
        results        : (B,)
    """
    board_tensors, legal_moves_list, target_indices, results = zip(*batch)

    # Stack board tensors — stessa dimensione per tutti
    board_tensors = torch.stack(board_tensors, dim=0)  # (B, 13, 8, 8)

    # Padding delle mosse legali
    # pad_sequence si aspetta (N, 46) per ogni elemento, batch_first=True → (B, N_max, 46)
    legal_moves_padded = pad_sequence(
        legal_moves_list,
        batch_first=True,
        padding_value=0.0
    )  # (B, N_max, 46)

    # Attention mask: True dove c'è una mossa reale, False dove c'è padding
    n_max = legal_moves_padded.shape[1]
    attention_mask = torch.zeros(len(batch), n_max, dtype=torch.bool)
    for i, moves in enumerate(legal_moves_list):
        attention_mask[i, :len(moves)] = True  # (B, N_max)

    target_indices = torch.tensor(target_indices, dtype=torch.long)   # (B,)
    results        = torch.tensor(results,        dtype=torch.float32) # (B,)

    return board_tensors, legal_moves_padded, attention_mask, target_indices, results


# ---------------------------------------------------------------------------
# Funzione di creazione dataloaders
# ---------------------------------------------------------------------------

def create_dataloaders(
    csv_file:    str  = "over_mate_1_tactic_evals.csv",
    batch_size:  int  = 128,
    num_workers: int  = 0,
    pin_memory:  bool = False,
):
    """
    Crea i DataLoader per train / validation / test.

    Il CSV deve avere le colonne: FEN, Move, Evaluation.

    Returns:
        trainloader, validationloader, testloader
    """
    trainset      = PointerChessDataset(csv_file, split='train')
    validationset = PointerChessDataset(csv_file, split='validation')
    testset       = PointerChessDataset(csv_file, split='test')

    g = torch.Generator()

    common = dict(
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, **common
    )
    validationloader = torch.utils.data.DataLoader(
        validationset, batch_size=batch_size, shuffle=False, **common
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, **common
    )

    print(f"Train set size:      {len(trainset)}")
    print(f"Validation set size: {len(validationset)}")
    print(f"Test set size:       {len(testset)}")

    return trainloader, validationloader, testloader