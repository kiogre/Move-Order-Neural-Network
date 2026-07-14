"""
lichess_eval_to_csv.py — Converte lichess_db_eval.jsonl.zst in CSV di training.

Formato input (JSONL, una riga per posizione):
    {"fen": "...", "evals": [{"depth": N, "knodes": K, "pvs": [{"cp": X, "mate": null, "line": "e2e4 ..."}]}]}

Formato output CSV (compatibile con train_stockfish.py):
    FEN, Evaluation, Move
    - Evaluation: centipawn come '+150', '-80', o mate '#+2', '#-3'
    - Move: prima mossa della PV migliore (mossa migliore Stockfish)

Utilizzo:
    # Step 1: conta posizioni per depth (veloce, nessuna scrittura)
    python lichess_eval_to_csv.py --input lichess_db_eval.jsonl.zst --stats-only

    # Step 2: converti con depth minima scelta
    python lichess_eval_to_csv.py --input lichess_db_eval.jsonl.zst --output stockfish_lichess.csv --min-depth 20

    # Opzionale: limita numero di posizioni output
    python lichess_eval_to_csv.py --input lichess_db_eval.jsonl.zst --output stockfish_lichess.csv --min-depth 20 --max-positions 5000000

Dipendenze:
    pip install zstandard tqdm
"""

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

import zstandard as zstd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Lettura file (zst o jsonl plain)
# ---------------------------------------------------------------------------

def open_jsonl(path: str):
    """Apre un .jsonl o .jsonl.zst in streaming, restituisce file-like di testo."""
    if path.endswith(".zst"):
        fh     = open(path, "rb")
        dctx   = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    else:
        return open(path, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Parsing di una riga JSONL
# ---------------------------------------------------------------------------

def best_eval(evals: list, min_depth: int):
    """
    Tra tutte le eval disponibili per una posizione, prende quella con
    depth massima >= min_depth.

    Restituisce (depth, cp_or_none, mate_or_none, best_move_uci) oppure None.
    """
    best = None
    for e in evals:
        depth = e.get("depth", 0)
        if depth < min_depth:
            continue
        if best is None or depth > best["depth"]:
            pvs = e.get("pvs", [])
            if not pvs:
                continue
            pv       = pvs[0]              # prima PV = mossa migliore
            line     = pv.get("line", "")
            if not line:
                continue
            best_move = line.split()[0]    # prima mossa della variante
            best = {
                "depth": depth,
                "cp":    pv.get("cp"),
                "mate":  pv.get("mate"),
                "move":  best_move,
            }
    return best


def format_evaluation(cp, mate) -> str:
    """
    Formatta l'evaluation nel formato atteso da train_stockfish.py:
        cp   -> '+150' o '-80'
        mate -> '#+2' o '#-3'
    """
    if mate is not None:
        sign = "+" if mate > 0 else "-"
        return f"#{sign}{abs(mate)}"
    if cp is not None:
        sign = "+" if cp >= 0 else ""
        return f"{sign}{cp}"
    return "+0"


# ---------------------------------------------------------------------------
# Modalita STATS: conta posizioni per depth
# ---------------------------------------------------------------------------

def run_stats(path: str, max_lines: int = None):
    """
    Scansiona il file e conta quante posizioni hanno almeno una eval
    con depth >= soglia per varie soglie.
    Utile per scegliere min_depth prima di fare la conversione vera.
    """
    print(f"Scansione statistiche: {path}")
    print("(Questo puo richiedere qualche minuto sul file completo)\n")

    depth_counter = Counter()   # depth -> n posizioni con quella depth massima
    total = 0
    errors = 0

    with open_jsonl(path) as f:
        for line in tqdm(f, desc="Posizioni", unit=" pos"):
            line = line.strip()
            if not line:
                continue
            try:
                obj   = json.loads(line)
                evals = obj.get("evals", [])
                if not evals:
                    continue
                max_d = max(e.get("depth", 0) for e in evals)
                depth_counter[max_d] += 1
                total += 1
            except Exception:
                errors += 1
            if max_lines and total >= max_lines:
                break

    print(f"\nTotale posizioni lette: {total:,}  (errori: {errors:,})")
    print("\nDistribuzione depth massima disponibile per posizione:")
    print(f"{'Depth':>8}  {'Posizioni':>12}  {'Cumul. >= depth':>16}")
    print("-" * 42)

    sorted_depths = sorted(depth_counter.keys(), reverse=True)
    cumul = 0
    for d in sorted_depths:
        cumul += depth_counter[d]
        print(f"{d:>8}  {depth_counter[d]:>12,}  {cumul:>16,}")

    print("\nSuggerimento: scegli --min-depth in base a quante posizioni vuoi.")
    print("  depth >= 20 -> buon compromesso qualita/quantita")
    print("  depth >= 16 -> piu posizioni, qualita leggermente inferiore")


# ---------------------------------------------------------------------------
# Modalita CONVERSIONE
# ---------------------------------------------------------------------------

def run_convert(path: str, output: str, min_depth: int, max_positions: int = None):
    print(f"Conversione: {path}")
    print(f"Output:      {output}")
    print(f"Min depth:   {min_depth}")
    if max_positions:
        print(f"Max posizioni: {max_positions:,}")
    print()

    written  = 0
    skipped  = 0
    errors   = 0

    with open_jsonl(path) as f, \
         open(output, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(["FEN", "Evaluation", "Move"])

        pbar = tqdm(f, desc="Posizioni lette", unit=" pos")
        for line in pbar:
            line = line.strip()
            if not line:
                continue

            try:
                obj   = json.loads(line)
                fen   = obj.get("fen", "").strip()
                evals = obj.get("evals", [])

                if not fen or not evals:
                    skipped += 1
                    continue

                ev = best_eval(evals, min_depth)
                if ev is None:
                    skipped += 1
                    continue

                evaluation = format_evaluation(ev["cp"], ev["mate"])
                move       = ev["move"]

                writer.writerow([fen, evaluation, move])
                written += 1

            except Exception:
                errors += 1
                continue

            if written % 100_000 == 0 and written > 0:
                pbar.set_postfix({"scritte": f"{written:,}", "skip": f"{skipped:,}"})

            if max_positions and written >= max_positions:
                print(f"\nRaggiunto limite di {max_positions:,} posizioni.")
                break

        pbar.close()

    size_mb = Path(output).stat().st_size / 1e6
    print(f"\nPosizioni scritte: {written:,}")
    print(f"Posizioni skippate (depth < {min_depth} o dati mancanti): {skipped:,}")
    print(f"Errori parsing: {errors:,}")
    print(f"File output: {output}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Converte lichess_db_eval.jsonl(.zst) in CSV per train_stockfish.py"
    )
    parser.add_argument("--input",         required=True,
                        help="File .jsonl o .jsonl.zst da Lichess")
    parser.add_argument("--output",        default="stockfish_lichess.csv",
                        help="File CSV di output (ignorato con --stats-only)")
    parser.add_argument("--min-depth",     type=int, default=20,
                        help="Depth minima Stockfish accettata (default: 20)")
    parser.add_argument("--max-positions", type=int, default=None,
                        help="Numero massimo di posizioni da scrivere (default: tutto)")
    parser.add_argument("--stats-only",    action="store_true",
                        help="Solo statistiche sulla distribuzione depth, non scrive CSV")
    parser.add_argument("--stats-lines",   type=int, default=None,
                        help="Con --stats-only: scansiona solo le prime N righe (per test rapido)")

    args = parser.parse_args()

    if args.stats_only:
        run_stats(args.input, max_lines=args.stats_lines)
    else:
        run_convert(
            path          = args.input,
            output        = args.output,
            min_depth     = args.min_depth,
            max_positions = args.max_positions,
        )


if __name__ == "__main__":
    main()
