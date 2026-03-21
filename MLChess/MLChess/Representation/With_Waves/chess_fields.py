"""
chess_fields.py
---------------
Computes temporal influence fields for chess positions.

For each piece on the board, we compute phi_i(x) — a measure of how
strongly piece i "influences" square x — using a weighted BFS where
influence decays with distance: phi_i(x) = alpha^(tau_i(x)).

The aggregate fields are:
  Phi_white(x) = sum_i phi_i(x)   for white pieces
  Phi_black(x) = sum_j phi_j(x)   for black pieces
  C(x) = Phi_white(x) - Phi_black(x)   control field

Blocking is handled correctly: sliding pieces (R, B, Q) cannot pass
through any occupied square. Non-sliding pieces (N, K, P) jump.
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMPTY  = 0
# White pieces
wP, wN, wB, wR, wQ, wK = 1, 2, 3, 4, 5, 6
# Black pieces
bP, bN, bB, bR, bQ, bK = -1, -2, -3, -4, -5, -6

PIECE_SYMBOLS = {
    wP:'P', wN:'N', wB:'B', wR:'R', wQ:'Q', wK:'K',
    bP:'p', bN:'n', bB:'b', bR:'r', bQ:'q', bK:'k',
}

FEN_TO_INT = {v: k for k, v in PIECE_SYMBOLS.items()}

# Directions for sliding pieces: (dr, dc)
ROOK_DIRS   = [(1,0),(-1,0),(0,1),(0,-1)]
BISHOP_DIRS = [(1,1),(1,-1),(-1,1),(-1,-1)]
QUEEN_DIRS  = ROOK_DIRS + BISHOP_DIRS

KNIGHT_MOVES = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
KING_MOVES   = [(dr,dc) for dr in [-1,0,1] for dc in [-1,0,1] if (dr,dc)!=(0,0)]


# ---------------------------------------------------------------------------
# FEN parsing
# ---------------------------------------------------------------------------

def fen_to_board(fen: str) -> np.ndarray:
    """
    Parse a FEN string and return an 8x8 int array.
    Row 0 = rank 8 (black's back rank), Row 7 = rank 1 (white's back rank).
    """
    board = np.zeros((8, 8), dtype=np.int8)
    position_part = fen.split()[0]
    rows = position_part.split('/')
    for r, row_str in enumerate(rows):
        c = 0
        for ch in row_str:
            if ch.isdigit():
                c += int(ch)
            else:
                board[r, c] = FEN_TO_INT[ch]
                c += 1
    return board


# ---------------------------------------------------------------------------
# BFS helpers
# ---------------------------------------------------------------------------

def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def bfs_distances(board: np.ndarray, r0: int, c0: int, piece: int) -> np.ndarray:
    """
    BFS on the actual board to compute the minimum number of moves for
    `piece` at (r0, c0) to reach every square.

    Rules:
    - Sliding pieces (R, B, Q) are blocked by any occupied square.
      They *can* land on an enemy square (capturing) but cannot pass through.
    - Non-sliding pieces (N, K) jump freely but cannot land on own pieces.
    - Pawns: move forward (or diagonally to capture), colour-aware.
      Here we compute *movement* reach — the squares where they can go,
      treating diagonal captures as reachable if an enemy is there OR we
      want to model the threat regardless.

    Returns an 8x8 array of distances (np.inf where unreachable).
    """
    dist = np.full((8, 8), np.inf)
    dist[r0, c0] = 0
    q = deque([(r0, c0, 0)])

    color = 1 if piece > 0 else -1   # +1 white, -1 black
    abs_piece = abs(piece)

    # For BFS on sliding pieces we use a visited array
    visited = np.zeros((8, 8), dtype=bool)
    visited[r0, c0] = True

    while q:
        r, c, d = q.popleft()

        if abs_piece == wN:   # Knight
            for dr, dc in KNIGHT_MOVES:
                nr, nc = r+dr, c+dc
                if _in_bounds(nr, nc) and not visited[nr, nc]:
                    # Can land if empty or enemy
                    if board[nr, nc] * color <= 0:
                        visited[nr, nc] = True
                        dist[nr, nc] = d + 1
                        q.append((nr, nc, d+1))

        elif abs_piece == wK:  # King
            for dr, dc in KING_MOVES:
                nr, nc = r+dr, c+dc
                if _in_bounds(nr, nc) and not visited[nr, nc]:
                    if board[nr, nc] * color <= 0:
                        visited[nr, nc] = True
                        dist[nr, nc] = d + 1
                        q.append((nr, nc, d+1))

        elif abs_piece == wP:  # Pawn — model threat squares
            # Forward direction depends on color
            fwd = -1 if color == 1 else 1   # white moves up (row-1), black down
            # Diagonal threats (always reachable for influence purposes)
            for dc in [-1, 1]:
                nr, nc = r+fwd, c+dc
                if _in_bounds(nr, nc) and not visited[nr, nc]:
                    visited[nr, nc] = True
                    dist[nr, nc] = d + 1
                    q.append((nr, nc, d+1))
            # Forward push (only if square is empty — cannot capture forward)
            nr, nc = r+fwd, c
            if _in_bounds(nr, nc) and not visited[nr, nc] and board[nr, nc] == EMPTY:
                visited[nr, nc] = True
                dist[nr, nc] = d + 1
                q.append((nr, nc, d+1))
                # Double push from starting rank
                start_rank = 6 if color == 1 else 1
                if r == start_rank:
                    nr2, nc2 = r+2*fwd, c
                    if _in_bounds(nr2, nc2) and not visited[nr2, nc2] and board[nr2, nc2] == EMPTY:
                        visited[nr2, nc2] = True
                        dist[nr2, nc2] = d + 1   # still 1 move away from start
                        q.append((nr2, nc2, d+1))

        else:  # Sliding: R, B, Q
            dirs = []
            if abs_piece in (wR, wQ): dirs += ROOK_DIRS
            if abs_piece in (wB, wQ): dirs += BISHOP_DIRS

            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                while _in_bounds(nr, nc):
                    if not visited[nr, nc]:
                        visited[nr, nc] = True
                        dist[nr, nc] = d + 1
                        # Can land if empty or enemy — but cannot continue past occupied
                        if board[nr, nc] != EMPTY:
                            if board[nr, nc] * color < 0:
                                # Enemy: can capture but ray stops
                                q.append((nr, nc, d+1))
                            # Own piece: ray stops, don't add to queue
                            break
                        else:
                            q.append((nr, nc, d+1))
                    else:
                        break
                    nr += dr
                    nc += dc

    return dist


# ---------------------------------------------------------------------------
# Field computation
# ---------------------------------------------------------------------------

def compute_piece_influence(dist: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    phi_i(x) = alpha^tau_i(x)
    Squares at distance 0 get influence 1.0 (the piece itself).
    Unreachable squares get 0.
    """
    phi = np.where(np.isinf(dist), 0.0, alpha ** dist)
    return phi


def compute_fields(
    board: np.ndarray,
    alpha: float = 0.5,
    return_per_piece: bool = False,
) -> dict:
    """
    Compute influence fields for all pieces on the board.

    Returns a dict with:
      'white'     : (8,8) aggregate influence field for white
      'black'     : (8,8) aggregate influence field for black
      'control'   : (8,8) C(x) = white - black
      'per_piece' : list of (r, c, piece, phi) if return_per_piece=True
    """
    white_field = np.zeros((8, 8))
    black_field = np.zeros((8, 8))
    per_piece = []

    for r in range(8):
        for c in range(8):
            piece = board[r, c]
            if piece == EMPTY:
                continue

            dist = bfs_distances(board, r, c, piece)
            phi  = compute_piece_influence(dist, alpha)

            if piece > 0:
                white_field += phi
            else:
                black_field += phi

            if return_per_piece:
                per_piece.append((r, c, piece, phi))

    result = {
        'white':   white_field,
        'black':   black_field,
        'control': white_field - black_field,
    }
    if return_per_piece:
        result['per_piece'] = per_piece
    return result


def compute_fields_from_fen(
    fen: str,
    alpha: float = 0.5,
    return_per_piece: bool = False,
) -> dict:
    board = fen_to_board(fen)
    fields = compute_fields(board, alpha=alpha, return_per_piece=return_per_piece)
    fields['board'] = board
    return fields


# ---------------------------------------------------------------------------
# Batch processing (for 2.5M positions)
# ---------------------------------------------------------------------------

def compute_fields_batch(
    fens: list[str],
    alpha: float = 0.5,
    flatten: bool = True,
) -> np.ndarray:
    """
    Process a list of FEN strings and return a numpy array of control fields.

    If flatten=True:  output shape (N, 64)  — ready for ML input
    If flatten=False: output shape (N, 8, 8)
    """
    N = len(fens)
    shape = (N, 64) if flatten else (N, 8, 8)
    out = np.zeros(shape, dtype=np.float32)

    for i, fen in enumerate(fens):
        try:
            fields = compute_fields_from_fen(fen, alpha=alpha)
            c = fields['control'].astype(np.float32)
            out[i] = c.flatten() if flatten else c
        except Exception as e:
            # Leave zeros for malformed FENs, optionally log
            pass

    return out


def compute_full_features_batch(
    fens: list[str],
    alpha: float = 0.5,
    flatten: bool = True,
) -> np.ndarray:
    """
    Returns 3 channels per position: [white_field, black_field, control_field]
    Shape: (N, 3, 64) if flatten else (N, 3, 8, 8)
    Useful as multi-channel input to a CNN.
    """
    N = len(fens)
    if flatten:
        out = np.zeros((N, 3, 64), dtype=np.float32)
    else:
        out = np.zeros((N, 3, 8, 8), dtype=np.float32)

    for i, fen in enumerate(fens):
        try:
            fields = compute_fields_from_fen(fen, alpha=alpha)
            for ch, key in enumerate(['white', 'black', 'control']):
                arr = fields[key].astype(np.float32)
                out[i, ch] = arr.flatten() if flatten else arr
        except Exception:
            pass

    return out


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

'''
if __name__ == '__main__':
    STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    fields = compute_fields_from_fen(STARTING_FEN, alpha=0.5)
    print("Control field (starting position):")
    print(np.round(fields['control'], 2))
    print("\nShould be ~symmetric (close to zero everywhere).")
    '''
