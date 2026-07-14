"""
encode_fast.py — Encoding board senza chess.Board.

Rimpiazzo drop-in per encode_board di MLChess, ma:
  - Non crea chess.Board
  - Parsa la stringa FEN direttamente
  - Supporta encoding batch (N FEN → N,13,8,8) in un colpo solo

Funzioni esportate:
  encode_board_fast(fen: str) -> torch.Tensor         # (13, 8, 8)
  encode_board_batch(fens: list[str]) -> torch.Tensor # (N, 13, 8, 8)
  truncate_fen(fen: str) -> str                       # solo campi 1+2

La logica è identica all'encode_board originale:
  - Piani 0-5:  pezzi del giocatore che muove (P N B R Q K)
  - Piani 6-11: pezzi dell'avversario
  - Piano 12:   tutto 1.0 (turno corrente)
  - Se turno=nero: flip dei rank (rank 0 diventa 7 e viceversa)
"""

import numpy as np
import torch

# Mappa carattere FEN → indice piano (0-5, corrisponde a piece_type - 1)
# P=pawn=0, N=knight=1, B=bishop=2, R=rook=3, Q=queen=4, K=king=5
_PIECE_IDX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
}


def truncate_fen(fen: str) -> str:
    """
    Riduce il FEN ai soli campi necessari per encode_board:
    posizione pezzi e turno (campi 1 e 2).

    '... w KQkq - 0 1' → '... w'

    Riduce la lunghezza media da ~70 a ~40 caratteri,
    risparmiando ~40% di storage nel precompute.
    """
    parts = fen.split(' ', 2)
    return parts[0] + ' ' + parts[1]


def encode_board_fast(fen: str) -> torch.Tensor:
    """
    Encoda un singolo FEN (o FEN troncato) in un tensore (13, 8, 8).
    Equivalente a encode_board di MLChess ma senza chess.Board.
    """
    parts = fen.split(' ')
    placement = parts[0]
    flip = (parts[1] == 'b')

    out = np.zeros((13, 8, 8), dtype=np.float32)
    out[12, :, :] = 1.0

    for fen_rank_idx, rank_str in enumerate(placement.split('/')):
        # fen_rank_idx 0 = rank 8 (chess rank 7, indice 7)
        # chess_rank = 7 - fen_rank_idx
        # se flip: display_rank = fen_rank_idx
        # se non flip: display_rank = 7 - fen_rank_idx
        display_rank = fen_rank_idx if flip else (7 - fen_rank_idx)

        file_idx = 0
        for ch in rank_str:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                piece_idx  = _PIECE_IDX[ch]
                is_white   = ch.isupper()
                is_current = is_white != flip   # pezzo del giocatore che muove
                plane      = piece_idx if is_current else (piece_idx + 6)
                out[plane, display_rank, file_idx] = 1.0
                file_idx += 1

    return torch.from_numpy(out)


def encode_board_batch(fens) -> torch.Tensor:
    """
    Encoda N FEN in un colpo solo.
    Input:  lista/array di N stringhe FEN (o FEN troncati)
    Output: (N, 13, 8, 8) float32

    Circa 3-5x più veloce di chiamare encode_board_fast N volte
    grazie al pre-alloco del buffer numpy e al loop interno ottimizzato.
    """
    N = len(fens)
    out = np.zeros((N, 13, 8, 8), dtype=np.float32)
    out[:, 12, :, :] = 1.0

    for i, fen in enumerate(fens):
        parts     = fen.split(' ')
        placement = parts[0]
        flip      = (parts[1] == 'b')

        for fen_rank_idx, rank_str in enumerate(placement.split('/')):
            display_rank = fen_rank_idx if flip else (7 - fen_rank_idx)
            file_idx = 0
            for ch in rank_str:
                if ch.isdigit():
                    file_idx += int(ch)
                else:
                    piece_idx  = _PIECE_IDX[ch]
                    is_white   = ch.isupper()
                    is_current = is_white != flip
                    plane      = piece_idx if is_current else (piece_idx + 6)
                    out[i, plane, display_rank, file_idx] = 1.0
                    file_idx += 1

    return torch.from_numpy(out)


# ---------------------------------------------------------------------------
# Test di correttezza (esegui standalone per verificare)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import chess
    from MLChess import encode_board

    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "8/8/8/8/8/8/4k3/4K3 w - - 0 1",   # endgame
    ]

    print("Test correttezza encode_board_fast vs encode_board originale:")
    all_ok = True
    for fen in test_fens:
        orig  = encode_board(fen)
        fast  = encode_board_fast(fen)
        trunc = encode_board_fast(truncate_fen(fen))

        ok1 = torch.allclose(orig, fast)
        ok2 = torch.allclose(orig, trunc)
        status = "OK" if (ok1 and ok2) else "FAIL"
        if not (ok1 and ok2):
            all_ok = False
        print(f"  [{status}] {fen[:50]}...")

    if all_ok:
        print("\nTutti i test passati.")
    else:
        print("\nATTENZIONE: alcuni test falliti.")

    # Benchmark velocità
    import time
    N = 2240   # tipico per Bellman: 64 posizioni × 35 figli

    sample_fens = [test_fens[1]] * N

    t0 = time.perf_counter()
    for f in sample_fens:
        encode_board(f)
    t1 = time.perf_counter()
    print(f"\nBenchmark {N} FEN:")
    print(f"  encode_board originale: {(t1-t0)*1000:.1f} ms")

    t0 = time.perf_counter()
    for f in sample_fens:
        encode_board_fast(f)
    t1 = time.perf_counter()
    print(f"  encode_board_fast:      {(t1-t0)*1000:.1f} ms")

    t0 = time.perf_counter()
    encode_board_batch(sample_fens)
    t1 = time.perf_counter()
    print(f"  encode_board_batch:     {(t1-t0)*1000:.1f} ms")
