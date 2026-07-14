"""
precompute_child_fens.py — Salva su disco le FEN dei figli per ogni posizione.

Per ogni posizione nel CSV:
  - Genera tutte le mosse legali (board.push/pop — fatto UNA VOLTA SOLA)
  - Salva i FEN troncati dei figli (solo campi 1+2, ~40 chars)

Storage: N × MAX_CHILDREN × 42 bytes
  5M  posizioni → ~8.8GB
  10M posizioni → ~17.6GB
  20M posizioni → ~35GB

Durante il training NON si chiama mai più chess.Board o board.push/pop.
encode_board_fast legge direttamente le stringhe pre-salvate.

Utilizzo:
  python precompute_child_fens.py --csv stockfish_lichess.csv --output child_fens/
  python precompute_child_fens.py --csv stockfish_lichess.csv --output child_fens/ --max-positions 10000000

Dipendenze: numpy, pandas, tqdm, chess
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import chess

from encode_fast import truncate_fen

# Lunghezza massima FEN troncato (campo1 + spazio + campo2)
# Media reale ~38 chars, 50 è sicuro per tutti i casi
FEN_MAX_LEN  = 90
MAX_CHILDREN = 80    # branching massimo reale in chess: ~218 ma >99.9% sotto 80


def precompute(csv_path: str, output_dir: str, max_positions: int = None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Caricamento CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["FEN"]).reset_index(drop=True)
    print(f"  Posizioni totali: {len(df):,}")

    if max_positions and len(df) > max_positions:
        df = df.iloc[:max_positions].reset_index(drop=True)
        print(f"  Limitato a: {max_positions:,}")

    N = len(df)

    # ---------------------------------------------------------------------------
    # Alloca memmap
    # ---------------------------------------------------------------------------
    child_fens_path  = os.path.join(output_dir, "child_fens.npy")
    n_children_path  = os.path.join(output_dir, "n_children.npy")

    size_gb = N * MAX_CHILDREN * FEN_MAX_LEN / 1e9
    print(f"\nAllocazione memmap:")
    print(f"  child_fens:  {size_gb:.2f} GB  ({N} × {MAX_CHILDREN} × {FEN_MAX_LEN} bytes)")
    print(f"  n_children:  {N * 2 / 1e6:.1f} MB")

    # dtype 'S{FEN_MAX_LEN}': stringa di byte a lunghezza fissa
    child_fens_mm = np.lib.format.open_memmap(
        child_fens_path, mode="w+",
        dtype=f"S{FEN_MAX_LEN}",
        shape=(N, MAX_CHILDREN),
    )
    n_children_mm = np.lib.format.open_memmap(
        n_children_path, mode="w+",
        dtype=np.uint8,     # max 218 mosse legali, uint8 basta
        shape=(N,),
    )

    child_fens_mm[:] = b""
    n_children_mm[:] = 0

    # ---------------------------------------------------------------------------
    # Generazione figli
    # ---------------------------------------------------------------------------
    skipped   = 0
    truncated = 0

    print("\nGenerazione FEN figli...")
    for i in tqdm(range(N), unit=" pos"):
        fen = str(df.iloc[i]["FEN"])
        try:
            board      = chess.Board(fen)
            legal_list = list(board.legal_moves)
            n          = len(legal_list)

            if n == 0:
                skipped += 1
                continue

            n_store = min(n, MAX_CHILDREN)
            if n > MAX_CHILDREN:
                truncated += 1

            for j, move in enumerate(legal_list[:n_store]):
                board.push(move)
                child_fen_trunc = truncate_fen(board.fen())
                # Tronca a FEN_MAX_LEN se necessario (non dovrebbe succedere)
                child_fens_mm[i, j] = child_fen_trunc[:FEN_MAX_LEN].encode("ascii")
                board.pop()

            n_children_mm[i] = n_store

        except Exception:
            skipped += 1
            continue

        if i % 50_000 == 0 and i > 0:
            child_fens_mm.flush()
            n_children_mm.flush()

    child_fens_mm.flush()
    n_children_mm.flush()

    # ---------------------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------------------
    meta = {
        "n_positions":    N,
        "max_children":   MAX_CHILDREN,
        "fen_max_len":    FEN_MAX_LEN,
        "skipped":        int(skipped),
        "truncated":      int(truncated),
        "source_csv":     csv_path,
        "child_fens_path": child_fens_path,
        "n_children_path": n_children_path,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nCompletato:")
    print(f"  Posizioni processate: {N - skipped:,}")
    print(f"  Skippate:             {skipped:,}")
    print(f"  Troncate (>{MAX_CHILDREN} mosse): {truncated:,}")
    print(f"  Storage:              {size_gb:.2f} GB")
    print(f"  Output:               {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",           default="stockfish_lichess.csv")
    parser.add_argument("--output",        default="child_fens")
    parser.add_argument("--max-positions", type=int, default=None)
    args = parser.parse_args()
    precompute(args.csv, args.output, args.max_positions)


if __name__ == "__main__":
    main()
