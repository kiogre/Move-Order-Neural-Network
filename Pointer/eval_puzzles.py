"""
eval_puzzles.py — Probe di valutazione su puzzle Lichess.

Misura, su un campione FISSO e MAI usato per il training:
  1. Calibrazione del value head: frazione di posizioni puzzle in cui
     value_pred(board) > 0 (la posizione e' per definizione favorevole
     a chi deve muovere, quindi ci si aspetta value_pred > 0 quasi sempre).
  2. Solve rate "policy pura": frazione di puzzle in cui argmax(probs)
     coincide con la prima mossa della soluzione (no MCTS, una sola
     forward pass batched -> molto economico).
  3. Solve rate "MCTS": stessa cosa ma usando get_best_move() con poche
     simulazioni (piu' costoso, campione piu' piccolo).

Formato CSV atteso (puzzle Lichess standard):
    PuzzleId, FEN, Moves, Rating, Themes, ...

Convenzione "Moves": il FEN e' la posizione PRIMA della mossa
dell'avversario. moves[0] = mossa dell'avversario (da applicare per
arrivare alla posizione del puzzle). moves[1] = prima mossa della
soluzione (quella che valutiamo).

Uso tipico:
    from eval_puzzles import PuzzleEvaluator
    evaluator = PuzzleEvaluator("lichess_puzzles.csv", device=DEVICE)
    stats = evaluator.evaluate(model, mcts)
    print(stats)

Eseguito direttamente, fa una valutazione standalone su un checkpoint.
"""

import random
import chess
import torch
import pandas as pd
from typing import Optional

from MLChess import encode_board, encode_legal_moves, JellyFishPointer, BatchedPointerMCTS
from puzzle_split import is_probe_puzzle

MOVE_VECTOR_DIM = 46


class PuzzleEvaluator:
    """
    Carica un campione fisso (seed fissa) di puzzle Lichess filtrati per
    tema, da usare come held-out probe durante il training AlphaZero.
    """

    def __init__(
        self,
        csv_file:        str,
        device:          torch.device,
        n_samples:       int = 300,
        themes_regex:    str = "mateIn1|mateIn2|mateIn3",
        seed:            int = 12345,
    ):
        self.device = device

        df = pd.read_csv(csv_file)
        df = df.dropna(subset=["FEN", "Moves", "Themes"])

        # Partizione held-out: garantisce nessuna sovrapposizione con i
        # pool di training (CurriculumDataset2 / MixedBufferDataset in
        # train_alphazero_v3.py), indipendentemente dai temi richiesti qui.
        df = df[df["PuzzleId"].apply(is_probe_puzzle)]

        mask = df["Themes"].str.contains(themes_regex, na=False)
        df = df[mask]

        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=seed)

        self.df = df.reset_index(drop=True)

        # Pre-calcola: posizione del puzzle (dopo moves[0]) e mossa soluzione
        self.positions = []  # list[(fen_puzzle, target_move_uci)]
        for _, row in self.df.iterrows():
            try:
                moves = str(row["Moves"]).split()
                if len(moves) < 2:
                    continue
                board = chess.Board(row["FEN"])
                board.push(chess.Move.from_uci(moves[0]))
                target_move = moves[1]
                # Sanity: la mossa target deve essere legale nella posizione
                if chess.Move.from_uci(target_move) not in board.legal_moves:
                    continue
                self.positions.append((board.fen(), target_move))
            except Exception:
                continue

        print(f"  [eval_puzzles] Caricati {len(self.positions)} puzzle "
              f"(temi='{themes_regex}', richiesti {n_samples})")

    def __len__(self):
        return len(self.positions)

    # ------------------------------------------------------------------
    # Probe 1+2: value calibration + policy solve rate (batched, no MCTS)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_policy_value(self, model: JellyFishPointer, batch_size: int = 64) -> dict:
        model.eval()

        n_correct      = 0
        n_value_ok     = 0
        n_total        = 0
        value_sum      = 0.0

        for start in range(0, len(self.positions), batch_size):
            batch = self.positions[start:start + batch_size]

            board_tensors = []
            moves_tensors = []
            move_lists    = []
            for fen, _ in batch:
                board = chess.Board(fen)
                board_tensors.append(encode_board(fen))
                legal = list(board.legal_moves)
                move_lists.append(legal)
                moves_tensors.append(encode_legal_moves(board))

            max_n = max(m.shape[0] for m in moves_tensors)
            B     = len(batch)

            boards_t = torch.stack(board_tensors).to(self.device)
            moves_padded = torch.zeros(B, max_n, MOVE_VECTOR_DIM, device=self.device)
            move_mask    = torch.zeros(B, max_n, dtype=torch.bool, device=self.device)
            for i, m in enumerate(moves_tensors):
                n = m.shape[0]
                moves_padded[i, :n] = m.to(self.device)
                move_mask[i, :n]    = True

            _, probs, value_pred = model(boards_t, moves_padded, move_mask)

            for i, (fen, target_uci) in enumerate(batch):
                legal = move_lists[i]
                n     = len(legal)
                pred_idx  = probs[i, :n].argmax().item()
                pred_move = legal[pred_idx]

                if pred_move.uci() == target_uci:
                    n_correct += 1

                v = value_pred[i, 0].item()
                value_sum += v
                if v > 0:
                    n_value_ok += 1

                n_total += 1

        if n_total == 0:
            return {"n": 0, "policy_solve_rate": None,
                    "value_calibration_rate": None, "value_mean": None}

        return {
            "n":                       n_total,
            "policy_solve_rate":       n_correct / n_total,
            "value_calibration_rate":  n_value_ok / n_total,
            "value_mean":              value_sum / n_total,
        }

    # ------------------------------------------------------------------
    # Probe 3: solve rate con MCTS (piu' costoso, campione ridotto)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_mcts(self, mcts: BatchedPointerMCTS, n_samples: int = 50,
                       num_simulations: int = 50, seed: int = 999) -> dict:
        if len(self.positions) == 0:
            return {"n": 0, "mcts_solve_rate": None}

        rng = random.Random(seed)
        sample = self.positions if len(self.positions) <= n_samples \
                 else rng.sample(self.positions, n_samples)

        n_correct = 0
        for fen, target_uci in sample:
            board = chess.Board(fen)
            move  = mcts.get_best_move(board, num_simulations=num_simulations, temperature=0.0)
            if move.uci() == target_uci:
                n_correct += 1

        return {"n": len(sample), "mcts_solve_rate": n_correct / len(sample)}

    # ------------------------------------------------------------------
    # Comodo: tutto insieme
    # ------------------------------------------------------------------

    def evaluate(self, model: JellyFishPointer, mcts: Optional[BatchedPointerMCTS] = None,
                  mcts_n_samples: int = 50, mcts_num_simulations: int = 50) -> dict:
        stats = self.evaluate_policy_value(model)
        if mcts is not None:
            mcts_stats = self.evaluate_mcts(mcts, n_samples=mcts_n_samples,
                                             num_simulations=mcts_num_simulations)
            stats["mcts_solve_rate"] = mcts_stats["mcts_solve_rate"]
            stats["mcts_n"]          = mcts_stats["n"]
        return stats


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--puzzles",    type=str, required=True)
    parser.add_argument("--n_samples",  type=int, default=300)
    parser.add_argument("--themes",     type=str, default="mateIn1|mateIn2|mateIn3")
    parser.add_argument("--mcts_samples", type=int, default=50)
    parser.add_argument("--mcts_sims",    type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = JellyFishPointer().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    mcts = BatchedPointerMCTS(model, device, c_puct=2.5)

    evaluator = PuzzleEvaluator(args.puzzles, device, n_samples=args.n_samples, themes_regex=args.themes)
    stats = evaluator.evaluate(model, mcts, mcts_n_samples=args.mcts_samples,
                                mcts_num_simulations=args.mcts_sims)

    print(stats)
