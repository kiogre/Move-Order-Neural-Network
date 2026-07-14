"""
precompute_bellman_pool.py — Pre-calcola e salva su disco il pool Bellman.

Per ogni posizione campionata dal CSV:
  - Encoda il board padre: (13, 8, 8)
  - Encoda tutti i board figli: (n_moves, 13, 8, 8)

Salva come numpy memmap in una directory — caricabile istantaneamente
durante il training senza chiamare mai python-chess.

Storage stimato:
  POOL_SIZE=100K → ~6.7GB
  POOL_SIZE=300K → ~20GB
  POOL_SIZE=500K → ~33GB

Utilizzo:
  python precompute_bellman_pool.py --csv stockfish_lichess.csv --output bellman_pool/ --pool-size 300000

Dipendenze: numpy, pandas, tqdm, chess, e MLChess (encode_board)
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import chess

from MLChess import encode_board

# ---------------------------------------------------------------------------
# Configurazione default
# ---------------------------------------------------------------------------

DEFAULT_CSV       = "stockfish_lichess.csv"
DEFAULT_OUTPUT    = "bellman_pool"
DEFAULT_POOL_SIZE = 300_000
MAX_MOVES         = 80    # padding massimo: copre >99.9% delle posizioni reali
NUM_CHANNELS      = 13
BOARD_H           = 8
BOARD_W           = 8
NUM_WORKERS_HINT  = 4     # per info, il preprocessing è single-process

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def precompute_pool(csv_path: str, output_dir: str, pool_size: int):
    os.makedirs(output_dir, exist_ok=True)

    # Carica CSV e campiona
    print(f"Caricamento CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["FEN"])
    print(f"  Posizioni totali: {len(df):,}")

    if pool_size < len(df):
        df = df.sample(n=pool_size, random_state=42).reset_index(drop=True)
        print(f"  Campionamento a {pool_size:,}")
    else:
        df = df.reset_index(drop=True)
        pool_size = len(df)
        print(f"  Uso tutte le {pool_size:,} posizioni")

    N    = pool_size
    C, H, W = NUM_CHANNELS, BOARD_H, BOARD_W

    # ---------------------------------------------------------------------------
    # Alloca memmap
    # ---------------------------------------------------------------------------
    parent_path   = os.path.join(output_dir, "parent_boards.npy")
    children_path = os.path.join(output_dir, "children_boards.npy")
    mask_path     = os.path.join(output_dir, "child_mask.npy")

    print(f"\nAllocazione memmap:")
    parent_size_gb   = N * C * H * W * 2 / 1e9
    children_size_gb = N * MAX_MOVES * C * H * W * 2 / 1e9
    print(f"  parent_boards:   {parent_size_gb:.2f} GB")
    print(f"  children_boards: {children_size_gb:.2f} GB")
    print(f"  child_mask:      {N * MAX_MOVES / 1e6:.1f} MB")
    print(f"  Totale stimato:  {parent_size_gb + children_size_gb:.2f} GB\n")

    parent_mm   = np.lib.format.open_memmap(
        parent_path, mode="w+", dtype=np.float16, shape=(N, C, H, W)
    )
    children_mm = np.lib.format.open_memmap(
        children_path, mode="w+", dtype=np.float16, shape=(N, MAX_MOVES, C, H, W)
    )
    mask_mm = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype=np.bool_, shape=(N, MAX_MOVES)
    )

    # Inizializza a zero (parent) e False (mask)
    parent_mm[:]   = 0
    children_mm[:] = 0
    mask_mm[:]     = False

    # ---------------------------------------------------------------------------
    # Encoding
    # ---------------------------------------------------------------------------
    skipped   = 0
    truncated = 0

    print("Encoding posizioni...")
    for i in tqdm(range(N), unit=" pos"):
        fen = str(df.iloc[i]["FEN"])
        try:
            board      = chess.Board(fen)
            legal_list = list(board.legal_moves)
            n_moves    = len(legal_list)

            if n_moves == 0:
                skipped += 1
                continue

            # Parent board
            parent_t = encode_board(fen)           # tensor (C, H, W)
            parent_mm[i] = parent_t.numpy().astype(np.float16)

            # Children boards
            n_store = min(n_moves, MAX_MOVES)
            if n_moves > MAX_MOVES:
                truncated += 1

            for j, move in enumerate(legal_list[:n_store]):
                board.push(move)
                child_t = encode_board(board.fen())
                children_mm[i, j] = child_t.numpy().astype(np.float16)
                mask_mm[i, j]     = True
                board.pop()

        except Exception as e:
            skipped += 1
            continue

        # Flush periodico per evitare di perdere tutto in caso di crash
        if i % 10_000 == 0 and i > 0:
            parent_mm.flush()
            children_mm.flush()
            mask_mm.flush()

    parent_mm.flush()
    children_mm.flush()
    mask_mm.flush()

    # ---------------------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------------------
    meta = {
        "pool_size":    N,
        "max_moves":    MAX_MOVES,
        "n_channels":   C,
        "board_h":      H,
        "board_w":      W,
        "skipped":      skipped,
        "truncated":    truncated,
        "source_csv":   csv_path,
        "parent_path":  parent_path,
        "children_path":children_path,
        "mask_path":    mask_path,
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nCompletato:")
    print(f"  Posizioni encodate: {N - skipped:,}")
    print(f"  Skippate (errori/terminali): {skipped:,}")
    print(f"  Troncate (> {MAX_MOVES} mosse): {truncated:,}")
    print(f"  Output: {output_dir}/")
    print(f"    parent_boards.npy   ({parent_size_gb:.2f} GB)")
    print(f"    children_boards.npy ({children_size_gb:.2f} GB)")
    print(f"    child_mask.npy")
    print(f"    meta.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pre-calcola pool Bellman su disco")
    parser.add_argument("--csv",       default=DEFAULT_CSV)
    parser.add_argument("--output",    default=DEFAULT_OUTPUT)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    args = parser.parse_args()

    precompute_pool(args.csv, args.output, args.pool_size)


if __name__ == "__main__":
    main()
