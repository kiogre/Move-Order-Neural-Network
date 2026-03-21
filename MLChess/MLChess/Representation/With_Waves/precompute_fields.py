"""
precompute_fields.py
---------------------
Precomputes influence fields for all positions in a CSV and saves them
to an HDF5 file. Run this ONCE before training.

This avoids recomputing BFS at every training step.

Output HDF5 structure:
  /white_field   (N, 8, 8)  float32  — white influence Φ_w(x)
  /black_field   (N, 8, 8)  float32  — black influence Φ_b(x)
  /control_field (N, 8, 8)  float32  — C(x) = Φ_w - Φ_b

Usage:
  python precompute_fields.py --csv your_dataset.csv --out fields.h5 --alpha 0.5 --workers 8
"""

import argparse
import time
import numpy as np
import h5py
import pandas as pd
from multiprocessing import Pool, cpu_count
from .chess_fields import compute_fields_from_fen


# ---------------------------------------------------------------------------
# Worker function (must be top-level for multiprocessing)
# ---------------------------------------------------------------------------

_ALPHA = 0.5   # set by main before spawning workers

def _process_fen(fen: str):
    """Compute fields for one FEN. Returns (white, black, control) as (8,8) float32."""
    try:
        fields = compute_fields_from_fen(fen, alpha=_ALPHA)
        return (
            fields['white'].astype(np.float32),
            fields['black'].astype(np.float32),
            fields['control'].astype(np.float32),
        )
    except Exception:
        return (
            np.zeros((8, 8), dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32),
        )


def _init_worker(alpha: float):
    global _ALPHA
    _ALPHA = alpha


# ---------------------------------------------------------------------------
# Main precompute function
# ---------------------------------------------------------------------------

def precompute(
    csv_path:   str,
    h5_path:    str,
    alpha:      float = 0.5,
    workers:    int   = None,
    chunk_size: int   = 2048,
):
    if workers is None:
        workers = min(cpu_count(), 8)

    print(f"Loading CSV: {csv_path}")
    df   = pd.read_csv(csv_path)
    fens = df['FEN'].tolist()
    N    = len(fens)
    print(f"  {N:,} positions  |  workers={workers}  |  alpha={alpha}")

    with h5py.File(h5_path, 'w') as h5:
        white_ds   = h5.create_dataset('white_field',   (N, 8, 8), dtype='float32',
                                        chunks=(chunk_size, 8, 8), compression='lzf')
        black_ds   = h5.create_dataset('black_field',   (N, 8, 8), dtype='float32',
                                        chunks=(chunk_size, 8, 8), compression='lzf')
        control_ds = h5.create_dataset('control_field', (N, 8, 8), dtype='float32',
                                        chunks=(chunk_size, 8, 8), compression='lzf')

        t0      = time.time()
        written = 0

        with Pool(workers, initializer=_init_worker, initargs=(alpha,)) as pool:
            # Process in chunks to keep memory bounded
            for start in range(0, N, chunk_size):
                batch_fens = fens[start : start + chunk_size]
                results    = pool.map(_process_fen, batch_fens)

                end = start + len(results)
                white_ds  [start:end] = np.stack([r[0] for r in results])
                black_ds  [start:end] = np.stack([r[1] for r in results])
                control_ds[start:end] = np.stack([r[2] for r in results])

                written += len(results)
                elapsed  = time.time() - t0
                rate     = written / elapsed
                eta      = (N - written) / rate if rate > 0 else 0

                print(f"  {written:>8,} / {N:,}  |  "
                      f"{rate:>6.0f} pos/s  |  "
                      f"ETA {eta/60:.1f} min", end='\r')

    print(f"\n✓ Saved to {h5_path}  ({written:,} positions)")
    elapsed = time.time() - t0
    print(f"  Total time: {elapsed/60:.1f} min  |  "
          f"Average: {N/elapsed:.0f} pos/s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute chess influence fields')
    parser.add_argument('--csv',     required=True,       help='Input CSV file')
    parser.add_argument('--out',     required=True,       help='Output HDF5 file')
    parser.add_argument('--alpha',   type=float, default=0.5,          help='Decay parameter (default: 0.5)')
    parser.add_argument('--workers', type=int,   default=None,         help='Number of CPU workers (default: auto)')
    parser.add_argument('--chunk',   type=int,   default=2048,         help='Chunk size for HDF5 writes')
    args = parser.parse_args()

    precompute(
        csv_path   = args.csv,
        h5_path    = args.out,
        alpha      = args.alpha,
        workers    = args.workers,
        chunk_size = args.chunk,
    )
'''
