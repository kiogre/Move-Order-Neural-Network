"""
pointer_graph_dataset.py
------------------------
Dataset per pointer network su scacchi con board codificata come grafo.

La scacchiera è rappresentata esattamente come in ChessLazyDenseDataset:
  - 64 nodi (una per casella)
  - feature nodo: one-hot(12) + valore pezzo + coord (x,y) → dim 14
  - edge_index fisso: tutte le direzioni regina + salti cavallo su ogni casella
    (grafo denso, NON filtrato sulle mosse legali)

Il resto dell'interfaccia è identico a PointerChessDataset:
  - legal_moves   : (N, 46)  encoding delle mosse legali
  - target_idx    : int      indice della mossa target
  - result        : float    valutazione normalizzata in [-1, 1]
"""

import chess
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Costanti (identiche a pointer_dataset.py)
# ---------------------------------------------------------------------------

PIECE_TYPE_TO_IDX = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4,  chess.KING: 5,
}

PROMO_TO_IDX = {
    None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4,
}

MOVE_VECTOR_DIM = 46

# ---------------------------------------------------------------------------
# Lookup tables per la rappresentazione a grafo (da new_graph_representation.py)
# ---------------------------------------------------------------------------

PIECE_TO_ID = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
}


def _precompute_queen_knight_edges() -> torch.Tensor:
    """Restituisce edge_index (2, E) con tutte le mosse regina+cavallo
    da ogni casella (grafo fisso, ~1900 archi)."""
    edges = []
    for sq in chess.SQUARES:
        r = chess.square_rank(sq)
        f = chess.square_file(sq)

        # Cavallo
        for dr, df in [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]:
            rr, ff = r + dr, f + df
            if 0 <= rr < 8 and 0 <= ff < 8:
                edges.append((sq, chess.square(ff, rr)))

        # Regina (8 direzioni, scivola fino al bordo)
        for dr, df in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            rr, ff = r + dr, f + df
            while 0 <= rr < 8 and 0 <= ff < 8:
                edges.append((sq, chess.square(ff, rr)))
                rr += dr
                ff += df

    return torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)


def _build_piece_lut() -> torch.Tensor:
    """LUT (13, 13): riga 0 = casella vuota, righe 1-12 = pezzi.
    Colonne 0-11: one-hot pezzo, colonna 12: valore normalizzato."""
    lut = torch.zeros((13, 13), dtype=torch.float32)
    values = [0,
               0.1,  0.325,  0.3,  0.5,  0.9,  1.0,
              -0.1, -0.325, -0.3, -0.5, -0.9, -1.0]
    for i in range(1, 13):
        lut[i, i - 1] = 1.0
        lut[i, 12]    = values[i]
    return lut


def _build_square_coords() -> torch.Tensor:
    """Coordinate (file, rank) normalizzate in [0,1] per le 64 caselle. (64, 2)"""
    coords = []
    for sq in chess.SQUARES:
        coords.append([(sq % 8) / 7.0, (sq // 8) / 7.0])
    return torch.tensor(coords, dtype=torch.float32)


# Precomputa una sola volta al momento dell'import
_EDGE_INDEX  = _precompute_queen_knight_edges()   # (2, E)  su CPU
_PIECE_LUT   = _build_piece_lut()                 # (13, 13)
_SQUARE_COORDS = _build_square_coords()            # (64, 2)


# ---------------------------------------------------------------------------
# Encoding della board come grafo
# ---------------------------------------------------------------------------

def encode_board_graph(fen: str) -> Data:
    """
    Codifica una posizione FEN come grafo PyG.

    Nodi : 64 caselle
    Feature (x) per nodo : one-hot(12) + valore_pezzo + file_norm + rank_norm → dim 14
    edge_index : grafo fisso regina+cavallo (identico per ogni posizione)

    Returns:
        torch_geometric.data.Data con attributi x e edge_index.
        (y, y_policy, legal_edge_mask NON inclusi — gestiti dal Dataset)
    """
    board = chess.Board(fen)

    piece_ids = torch.zeros(64, dtype=torch.long)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            piece_ids[sq] = PIECE_TO_ID[p.symbol()]

    # x: (64, 13) → concat con coords → (64, 14+1) = (64, 14)
    # Nota: piece_lut ha 13 col (12 one-hot + 1 valore), coords ha 2 col → tot 14
    x = _PIECE_LUT[piece_ids]                   # (64, 13)
    x = torch.cat([x, _SQUARE_COORDS], dim=1)  # (64, 15) → 12 one-hot + val + file + rank

    return Data(x=x, edge_index=_EDGE_INDEX)


# ---------------------------------------------------------------------------
# Encoding mosse (identico a pointer_dataset.py)
# ---------------------------------------------------------------------------

def encode_move(move: chess.Move, board: chess.Board) -> torch.Tensor:
    vec = torch.zeros(MOVE_VECTOR_DIM, dtype=torch.float32)

    flip = board.turn == chess.BLACK

    from_row = chess.square_rank(move.from_square)
    from_col = chess.square_file(move.from_square)
    to_row   = chess.square_rank(move.to_square)
    to_col   = chess.square_file(move.to_square)

    if flip:
        from_row = 7 - from_row
        to_row   = 7 - to_row

    piece = board.piece_at(move.from_square)
    if piece is not None:
        vec[PIECE_TYPE_TO_IDX[piece.piece_type]] = 1.0

    vec[6  + from_row] = 1.0
    vec[14 + from_col] = 1.0
    vec[22 + to_row]   = 1.0
    vec[30 + to_col]   = 1.0

    vec[38] = 1.0 if board.is_capture(move)    else 0.0
    vec[39] = 1.0 if board.is_en_passant(move) else 0.0
    vec[40] = 1.0 if board.is_castling(move)   else 0.0

    promo_idx = PROMO_TO_IDX.get(move.promotion, 0)
    vec[41 + promo_idx] = 1.0

    return vec


def encode_legal_moves(board: chess.Board) -> torch.Tensor:
    """Restituisce (N, 46) con l'encoding di tutte le N mosse legali."""
    moves = list(board.legal_moves)
    if not moves:
        return torch.zeros((0, MOVE_VECTOR_DIM), dtype=torch.float32)
    return torch.stack([encode_move(m, board) for m in moves], dim=0)


def encode_result(result: str, max_cp: int = 1000) -> float:
    """Normalizza la valutazione in [-1, 1]."""
    if '#' in result:
        return 1.0 if '+' in result else -1.0
    return max(-max_cp, min(max_cp, int(result))) / max_cp


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PointerGraphChessDataset(torch.utils.data.Dataset):
    """
    Dataset per pointer network su scacchi.

    La board è codificata come grafo PyG (nodi=64, edge fissi regina+cavallo).
    Le mosse e il target sono identici a PointerChessDataset.

    Ogni elemento restituisce:
        graph        : torch_geometric.data.Data  con x (64,15) e edge_index
        legal_moves  : (N, 46)   encoding delle N mosse legali
        target_idx   : int       indice della mossa target in legal_moves
        result       : float     valutazione in [-1, 1]
    """

    def __init__(self, csv_file: str, split: str = 'train'):
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

        # Board → grafo
        graph = encode_board_graph(fen)

        # Mosse legali
        legal_moves_list   = list(board.legal_moves)
        legal_moves_tensor = encode_legal_moves(board)  # (N, 46)

        # Indice della mossa target
        target_move = chess.Move.from_uci(move_uci)
        try:
            target_idx = legal_moves_list.index(target_move)
        except ValueError:
            target_idx = 0  # fallback (dati sporchi)

        result = encode_result(result_str)

        return graph, legal_moves_tensor, target_idx, result


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn_pointer_graph(batch):
    """
    Combina un batch con numero variabile di mosse legali.

    Returns:
        graphs         : list[Data]        lista di grafi PyG (B elementi)
                         — usa torch_geometric.loader.DataLoader per batching
                           automatico; qui li lasciamo come lista per flessibilità
        legal_moves    : (B, N_max, 46)   paddato con zeri
        attention_mask : (B, N_max)        True = mossa reale, False = padding
        target_indices : (B,)
        results        : (B,)
    """
    graphs, legal_moves_list, target_indices, results = zip(*batch)

    # Padding mosse legali
    legal_moves_padded = pad_sequence(
        legal_moves_list, batch_first=True, padding_value=0.0
    )  # (B, N_max, 46)

    # Attention mask
    n_max = legal_moves_padded.shape[1]
    attention_mask = torch.zeros(len(batch), n_max, dtype=torch.bool)
    for i, moves in enumerate(legal_moves_list):
        attention_mask[i, :len(moves)] = True

    target_indices = torch.tensor(target_indices, dtype=torch.long)
    results        = torch.tensor(results,        dtype=torch.float32)

    return list(graphs), legal_moves_padded, attention_mask, target_indices, results


# ---------------------------------------------------------------------------
# Factory dataloaders
# ---------------------------------------------------------------------------

def create_dataloaders_pointer_graph(
    csv_file:    str  = "over_mate_1_tactic_evals.csv",
    batch_size:  int  = 128,
    num_workers: int  = 0,
    pin_memory:  bool = False,
):
    """
    Crea DataLoader train / validation / test.

    Nota: i grafi PyG sono restituiti come lista dal collate_fn.
    Se vuoi il batching automatico dei grafi (batch.x, batch.edge_index con batch vector),
    sostituisci torch.utils.data.DataLoader con torch_geometric.loader.DataLoader
    e rimuovi il collate_fn personalizzato.

    Returns:
        trainloader, validationloader, testloader
    """
    trainset      = PointerGraphChessDataset(csv_file, split='train')
    validationset = PointerGraphChessDataset(csv_file, split='validation')
    testset       = PointerGraphChessDataset(csv_file, split='test')

    g = torch.Generator()

    common = dict(
        collate_fn  = collate_fn_pointer_graph,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        generator   = g,
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
