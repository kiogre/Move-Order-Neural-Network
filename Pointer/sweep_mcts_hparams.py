"""
sweep_mcts_hparams.py — Trova batch_size / num_simulations giusti per
PointerMCTS dopo il fix della virtual loss.

I vecchi valori (batch_size=256, num_simulations=40000) erano tarati per
compensare simulazioni sprecate da un bug. Ora ogni simulazione fa lavoro
vero: probabilmente serve un ordine di grandezza in meno di simulazioni
per lo stesso solve rate. Questo script misura, non indovina.

Uso:
  python sweep_mcts_hparams.py --checkpoint checkpoints_pointer_20m_from_fast/best.pt --puzzles lichess_puzzles.csv
"""

import argparse
import torch

from MLChess import JellyFishPointer, PointerMCTS
from eval_puzzles import PuzzleEvaluator

BATCH_SIZES = [32]
SIM_COUNTS  = [400, 800, 1600, 3200]
C_PUCT_VALS = [1.0, 1.5, 2.0, 2.5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--puzzles",    type=str, required=True)
    parser.add_argument("--n_samples",  type=int, default=200)
    parser.add_argument("--mcts_samples", type=int, default=30)
    parser.add_argument("--themes",     type=str, default="mateIn1|mateIn2|mateIn3",
                         help="Regex sulla colonna Themes del CSV puzzle. "
                              "Vedi i nomi esatti in strings.xml (Lichess).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = JellyFishPointer().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    evaluator = PuzzleEvaluator(args.puzzles, device, n_samples=args.n_samples,
                                 themes_regex=args.themes)

    print(f"\n{'c_puct':>7} {'batch_size':>10} {'sims':>8} {'solve_rate':>11} {'avg_time_s':>11}")
    print("-" * 54)

    for cp in C_PUCT_VALS:
        for bs in BATCH_SIZES:
            mcts = PointerMCTS(model, device, c_puct=cp, batch_size=bs)
            for sims in SIM_COUNTS:
                stats = evaluator.evaluate_mcts(
                    mcts, n_samples=args.mcts_samples, num_simulations=sims
                )
                print(f"{cp:>7} {bs:>10} {sims:>8} {stats['mcts_solve_rate']:>11.3f} "
                      f"{stats['avg_time_s']:>11.3f}")


if __name__ == "__main__":
    main()
