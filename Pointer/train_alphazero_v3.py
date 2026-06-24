###### RICORDARE MODIFICARE PUNTO PARTENZA TRAINING

"""
train_alphazero_v3.py — Training AlphaZero-style per JellyFishPointer (BatchedPointerMCTS).

Differenze rispetto a train_alphazero_v2.py:
  - RIMOSSO evaluate_vs_frozen / winrate_vs_frozen come criterio di
    selezione del checkpoint: era "MCTS(main) vs greedy(main)", non
    "main vs snapshot precedente" (frozen_model non era mai usato per
    generare partite). Era quindi una metrica strutturalmente alta e
    poco informativa, oltre a costare ~100 partite extra per epoca.
  - AGGIUNTO probe periodico su puzzle Lichess (eval_puzzles.PuzzleEvaluator):
      * value_calibration_rate -> diagnostica diretta sul value head
      * policy_solve_rate      -> qualita' della policy pura
      * mcts_solve_rate        -> contributo della ricerca MCTS
    Eseguito ogni PUZZLE_EVAL_EVERY epoche, su un campione held-out
    (mai usato per il training). best.pt viene salvato in base a
    questo punteggio invece che al vecchio winrate.
  - Scheduler ora basato su (policy_loss + value_loss) (mode='min'),
    sempre disponibile ad ogni epoca, invece che su winrate.
  - CURRICULUM_PROB abbassato a 0.35 (prima 1.00 / 0.70): la maggior
    parte delle partite parte ancora da posizioni curriculum, ma una
    quota consistente parte dalla posizione iniziale standard.
  - MIXED_BUFFER_RATIO riattivato a 0.15: una piccola percentuale di
    campioni diretti dal dataset tattico (solo policy loss) per non
    perdere la "sharpness" tattica durante il fine-tuning RL.
  - Temi curriculum allargati (CurriculumDataset2).
  - frozen_model/frozen_mcts mantenuti SOLO per compatibilita' di
    checkpoint (caricamento di run precedenti) ma non piu' usati nel
    loop principale.
"""

import os
import math
import random
import pickle
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import deque
from tqdm import tqdm

from MLChess import encode_board, encode_legal_moves, JellyFishPointer, BatchedPointerMCTS
from eval_puzzles import PuzzleEvaluator
from puzzle_split import is_probe_puzzle

MOVE_VECTOR_DIM = 46

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

SUPERVISED_CHECKPOINT = "checkpoints_lichess/best.pt"
AZ_CHECKPOINT_DIR     = "checkpoints_az_v3"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS          = 500
GAMES_PER_EPOCH = 64
MAX_MOVES       = 400

# MCTS
NUM_SIMULATIONS = 400
TEMP_HIGH       = 1.0
TEMP_LOW        = 0.01
TEMP_THRESHOLD  = 10
C_PUCT          = 2.5

# Replay buffer
BUFFER_SIZE = 200_000
MIN_BUFFER  = 1_000

# Training
TRAIN_STEPS       = 200
BATCH_SIZE        = 256
LR_BACKBONE       = 5e-5
LR_HEADS          = 5e-5
VALUE_LOSS_WEIGHT = 3.0   # peso value loss per ricalibrazione scala predizioni

# Curriculum learning (posizioni di partenza per il self-play) e mixed buffer
# — entrambi nel formato puzzle Lichess (PuzzleId, FEN, Moves, Themes, ...).
# Possono puntare allo stesso file: CurriculumDataset2 e MixedBufferDataset
# usano filtri/regex sui temi leggermente diversi e campionano in modo
# indipendente, quindi va bene anche se e' lo stesso CSV.
CURRICULUM_CSV  = "lichess_db_puzzle.csv"
CURRICULUM_PROB = 0.35

# Mixed buffer — campioni diretti dal dataset tattico (solo policy loss)
MIXED_BUFFER_RATIO = 0.15
MIXED_BUFFER_SIZE  = 50_000

# Probe su puzzle Lichess (held-out, MAI usato per training/curriculum)
PUZZLE_CSV            = "lichess_db_puzzle.csv"
PUZZLE_EVAL_EVERY     = 5      # ogni quante epoche fare il probe
PUZZLE_EVAL_N         = 300    # numero di puzzle nel set held-out (policy+value)
PUZZLE_EVAL_THEMES    = "mateIn1|mateIn2|mateIn3"
PUZZLE_MCTS_SAMPLES   = 50      # sottocampione per il probe con MCTS (piu' costoso)
PUZZLE_MCTS_SIMS      = 50

# Score combinato per la selezione di best.pt:
#   peso uguale a calibrazione value + solve rate MCTS
def puzzle_score(stats: dict) -> float:
    vc = stats.get("value_calibration_rate") or 0.0
    ms = stats.get("mcts_solve_rate")
    if ms is None:
        ms = stats.get("policy_solve_rate") or 0.0
    return 0.5 * vc + 0.5 * ms


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model: JellyFishPointer) -> Adam:
    return Adam([
        {"params": list(model.backbone.parameters()) +
                   list(model.move_encoder.parameters()),
         "lr": LR_BACKBONE, "name": "backbone"},
        {"params": list(model.policy_head.parameters()) +
                   list(model.value_head.parameters()),
         "lr": LR_HEADS, "name": "heads"},
    ])


# ---------------------------------------------------------------------------
# Curriculum dataset — posizioni di partenza per il self-play
# ---------------------------------------------------------------------------

class CurriculumDataset2:
    """
    Carica posizioni da puzzle Lichess filtrati per tema, da usare come
    posizioni di partenza per il self-play (curriculum learning).

    Formato CSV Lichess puzzles:
        PuzzleId, FEN, Moves, Rating, Themes, ...

    Il FEN e' la posizione PRIMA della mossa dell'avversario.
    Applica la prima mossa per ottenere la posizione reale del puzzle.

    Esclude la partizione "probe" (puzzle_split.is_probe_puzzle), cosi'
    nessuna posizione di partenza per il self-play coincide con i puzzle
    held-out usati da PuzzleEvaluator — indipendentemente dai temi.
    """
    def __init__(self, csv_file: str, max_samples: int = MIXED_BUFFER_SIZE):
        tqdm.write(f"  Caricamento curriculum dataset da {csv_file}...")
        df = pd.read_csv(csv_file)
        df = df[~df["PuzzleId"].apply(is_probe_puzzle)]

        mask = (
            df['Themes'].str.contains(
                'advantage|crushing|endgame|middlegame|opening', na=False
            )
            & ~df['Themes'].str.contains('mateIn1', na=False)
        )
        df = df[mask].dropna(subset=["FEN", "Moves"])

        if len(df) > max_samples * 4:
            df = df.sample(n=max_samples * 4, random_state=42)

        # Pre-calcola i FEN reali applicando la prima mossa (mossa avversario)
        fens = []
        for _, row in df.iterrows():
            try:
                board = chess.Board(row["FEN"])
                first_move = chess.Move.from_uci(row["Moves"].split()[0])
                board.push(first_move)
                fens.append(board.fen())
            except Exception:
                continue

        self.fens     = fens
        self.max_size = max_samples
        tqdm.write(f"  Curriculum dataset: {len(self.fens)} posizioni")

    def get_random_fen(self) -> str:
        return self.fens[random.randint(0, len(self.fens) - 1)]


class MixedBufferDataset:
    """
    Dataset tattico per il "mixed buffer".

    Due pool separati, entrambi caricati una volta all'avvio e MAI soggetti
    a eviction (non sono nel replay_buffer, che invece e' un deque con
    maxlen=BUFFER_SIZE e fa FIFO sugli step di self-play):

      - self.positions: temi 'advantage|crushing|middlegame|opening'
        (mateIn1 escluso). policy_target = one-hot sulla mossa soluzione,
        value_target = None (mascherato dalla value loss, solo policy —
        comportamento invariato rispetto a prima).

      - self.tactical_positions: temi 'mateIn2|mateIn3' (mateIn1 escluso,
        riservato al probe held-out di eval_puzzles.py). policy_target =
        one-hot sulla prima mossa della soluzione, value_target = +1.0
        FISSO: per definizione, chi muove in un puzzle mateIn2/mateIn3 ha
        una vittoria forzata, quindi value(posizione) deve essere
        fortemente positivo. Inietta direttamente nella value loss un
        segnale di calibrazione su posizioni nettamente vincenti, senza
        dover aspettare che il self-play le incontri (raramente).

    TACTICAL_RATIO controlla quanta parte di ogni mini-batch mixed viene
    presa dal secondo pool.

    Entrambi i pool escludono la partizione "probe" (puzzle_split.
    is_probe_puzzle), quindi e' garantito zero overlap con i puzzle
    held-out usati da PuzzleEvaluator — indipendentemente dai temi.
    """
    TACTICAL_RATIO = 0.3

    def __init__(self, csv_file: str, max_samples: int = MIXED_BUFFER_SIZE,
                 themes_regex: str = "advantage|crushing|middlegame|opening",
                 tactical_themes_regex: str = "mateIn2|mateIn3"):
        tqdm.write(f"  Caricamento mixed buffer dataset da {csv_file}...")
        df = pd.read_csv(csv_file).dropna(subset=["FEN", "Moves", "Themes"])
        df = df[~df["PuzzleId"].apply(is_probe_puzzle)]

        general_mask = (
            df['Themes'].str.contains(themes_regex, na=False)
            & ~df['Themes'].str.contains('mateIn1', na=False)
        )
        general_df = df[general_mask]
        if len(general_df) > max_samples * 4:
            general_df = general_df.sample(n=max_samples * 4, random_state=43)
        self.positions = self._build_positions(general_df)

        tactical_mask = (
            df['Themes'].str.contains(tactical_themes_regex, na=False)
            & ~df['Themes'].str.contains('mateIn1', na=False)
        )
        tactical_df = df[tactical_mask]
        if len(tactical_df) > max_samples * 4:
            tactical_df = tactical_df.sample(n=max_samples * 4, random_state=44)
        self.tactical_positions = self._build_positions(tactical_df)

        tqdm.write(f"  Mixed buffer: {len(self.positions)} posizioni generali "
                   f"(value=None), {len(self.tactical_positions)} posizioni "
                   f"tattiche (value=+1.0)")

    @staticmethod
    def _build_positions(df) -> list[tuple[str, str]]:
        positions = []
        for _, row in df.iterrows():
            try:
                moves = str(row["Moves"]).split()
                if len(moves) < 2:
                    continue
                board = chess.Board(row["FEN"])
                board.push(chess.Move.from_uci(moves[0]))
                target_uci = moves[1]
                if chess.Move.from_uci(target_uci) not in board.legal_moves:
                    continue
                positions.append((board.fen(), target_uci))
            except Exception:
                continue
        return positions

    def get_mixed_samples(self, n: int) -> list[dict]:
        n_tactical = int(n * self.TACTICAL_RATIO) if self.tactical_positions else 0
        n_general  = n - n_tactical
        samples  = self._sample_from(self.positions, n_general, value_target=None)
        samples += self._sample_from(self.tactical_positions, n_tactical, value_target=1.0)
        return samples

    def _sample_from(self, pool: list[tuple[str, str]], n: int,
                      value_target: float | None) -> list[dict]:
        if not pool or n <= 0:
            return []
        sample = random.sample(pool, min(n, len(pool)))
        out = []
        for fen, target_uci in sample:
            try:
                board      = chess.Board(fen)
                legal_list = list(board.legal_moves)
                if not legal_list:
                    continue

                target_move   = chess.Move.from_uci(target_uci)
                legal_moves_t = encode_legal_moves(board)

                target_vec = torch.zeros(len(legal_list))
                if target_move in legal_list:
                    target_vec[legal_list.index(target_move)] = 1.0
                else:
                    target_vec[0] = 1.0

                out.append({
                    "board_fen":     fen,
                    "legal_moves":   legal_moves_t,
                    "policy_target": target_vec,
                    "value_target":  value_target,
                })
            except Exception:
                continue
        return out


# ---------------------------------------------------------------------------
# Training sul replay buffer
# ---------------------------------------------------------------------------

def train_on_buffer(
    model:         JellyFishPointer,
    optimizer:     Adam,
    buffer:        list[dict],
    n_steps:       int,
    device:        torch.device,
    mixed_ds:      MixedBufferDataset = None,
) -> dict:
    model.train()

    total_policy_loss = 0.0
    total_value_loss  = 0.0

    for _ in range(n_steps):
        n_mixed  = int(BATCH_SIZE * MIXED_BUFFER_RATIO) if mixed_ds else 0
        n_buffer = BATCH_SIZE - n_mixed

        batch = random.sample(buffer, min(n_buffer, len(buffer)))

        if n_mixed > 0 and mixed_ds:
            mixed = mixed_ds.get_mixed_samples(n_mixed)
            batch = batch + mixed

        board_tensors  = []
        moves_tensors  = []
        policy_targets = []
        value_targets  = []

        for step in batch:
            board_tensors.append(encode_board(step["board_fen"]))
            moves_tensors.append(step["legal_moves"])
            policy_targets.append(step["policy_target"])
            value_targets.append(step["value_target"])  # None per campioni mixed

        max_n = max(m.shape[0] for m in moves_tensors)
        B     = len(batch)

        moves_padded  = torch.zeros(B, max_n, MOVE_VECTOR_DIM, device=device)
        move_mask     = torch.zeros(B, max_n, dtype=torch.bool,  device=device)
        policy_padded = torch.zeros(B, max_n, device=device)

        for i, (m, p) in enumerate(zip(moves_tensors, policy_targets)):
            n = m.shape[0]
            moves_padded[i, :n]  = m.to(device)
            move_mask[i, :n]     = True
            policy_padded[i, :n] = p.to(device)

        boards_t = torch.stack(board_tensors).to(device)

        # Maschera value loss: escludi campioni mixed (value_target = None)
        value_mask = torch.tensor(
            [v is not None for v in value_targets],
            dtype=torch.bool, device=device
        )
        values_clean = [v if v is not None else 0.0 for v in value_targets]
        values_t = torch.tensor(values_clean, dtype=torch.float32, device=device).unsqueeze(1)

        # Forward
        logits, probs, value_pred = model(boards_t, moves_padded, move_mask)

        # Policy loss — cross-entropy con distribuzione soft MCTS / one-hot
        log_probs   = torch.log(probs + 1e-8)
        policy_loss = -(policy_padded * log_probs).sum(dim=1).mean()

        # Value loss — solo sui campioni MCTS (non mixed)
        if value_mask.any():
            value_loss = F.mse_loss(value_pred[value_mask], values_t[value_mask])
        else:
            value_loss = torch.tensor(0.0, device=device)

        loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss  += value_loss.item()

    model.eval()

    return {
        "policy_loss": total_policy_loss / n_steps,
        "value_loss":  total_value_loss  / n_steps,
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str, replay_buffer):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    buffer_path = path.replace(".pt", "_buffer.pkl")
    buffer_tmp  = buffer_path + ".tmp"
    with open(buffer_tmp, "wb") as f:
        pickle.dump(list(replay_buffer), f)
    os.replace(buffer_tmp, buffer_path)

    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)

    tqdm.write(f"  → checkpoint salvato: {path}")


def load_checkpoint(path, model, optimizer, scheduler, replay_buffer):
    ckpt = torch.load(path, map_location=DEVICE)

    state_dict = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt and scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    epoch     = ckpt.get("epoch", 0)
    best_score = ckpt.get("best_score", 0.0)
    frozen_sd = ckpt.get("frozen_state_dict", None)  # solo per compatibilita'

    tqdm.write(f"  → checkpoint caricato: {path}  (epoch {epoch}, best score {best_score:.3f})")

    buffer_path = path.replace(".pt", "_buffer.pkl")
    if os.path.exists(buffer_path):
        with open(buffer_path, "rb") as f:
            loaded = pickle.load(f)
        replay_buffer.extend(loaded)
        tqdm.write(f"  → buffer caricato: {len(replay_buffer)} step")

    return epoch, best_score, frozen_sd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    main_model = JellyFishPointer().to(DEVICE)

    optimizer = build_optimizer(main_model)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=40)

    start_epoch = 1
    best_score  = 0.0
    replay_buffer: deque[dict] = deque(maxlen=BUFFER_SIZE)

    az_last = os.path.join(AZ_CHECKPOINT_DIR, "last.pt")
    az_best = os.path.join(AZ_CHECKPOINT_DIR, "best.pt")

    if os.path.exists(az_best):
        print("Checkpoint AlphaZero trovato, riprendo...")
        start_epoch, _, _ = load_checkpoint(
            az_best, main_model, optimizer, scheduler, replay_buffer
        )
        start_epoch += 1

        if os.path.exists(az_best):
            best_ckpt  = torch.load(az_best, map_location=DEVICE)
            best_score = best_ckpt.get("best_score", 0.0)
            tqdm.write(f"  → best score storico: {best_score:.3f}")

    elif os.path.exists(SUPERVISED_CHECKPOINT):
        print(f"Carico supervised: {SUPERVISED_CHECKPOINT}")
        ckpt = torch.load(SUPERVISED_CHECKPOINT, map_location=DEVICE)
        main_model.load_state_dict(ckpt["model"])
        buffer_path = SUPERVISED_CHECKPOINT.replace(".pt", "_buffer.pkl")
        if os.path.exists(buffer_path):
            with open(buffer_path, "rb") as f:
                loaded = pickle.load(f)
            replay_buffer.extend(loaded)
            tqdm.write(f"  → buffer caricato: {len(replay_buffer)} step")

    else:
        print("Nessun checkpoint trovato, parto da zero.")

    # Ripristina LR esplicitamente (override del checkpoint)
    for group in optimizer.param_groups:
        if group["name"] == "backbone":
            group["lr"] = LR_BACKBONE
        else:
            group["lr"] = LR_HEADS

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=40)

    main_model.eval()

    # Curriculum dataset (posizioni di partenza self-play)
    curriculum_ds = None
    if os.path.exists(CURRICULUM_CSV):
        curriculum_ds = CurriculumDataset2(CURRICULUM_CSV)
    else:
        tqdm.write(f"  WARNING: {CURRICULUM_CSV} non trovato, curriculum disabilitato")

    # Mixed buffer dataset (campioni diretti, solo policy loss)
    mixed_ds = None
    if MIXED_BUFFER_RATIO > 0 and os.path.exists(CURRICULUM_CSV):
        mixed_ds = MixedBufferDataset(CURRICULUM_CSV)
    elif MIXED_BUFFER_RATIO > 0:
        tqdm.write(f"  WARNING: {CURRICULUM_CSV} non trovato, mixed buffer disabilitato")

    # Probe su puzzle Lichess (held-out)
    puzzle_eval = None
    if os.path.exists(PUZZLE_CSV):
        puzzle_eval = PuzzleEvaluator(
            PUZZLE_CSV, DEVICE, n_samples=PUZZLE_EVAL_N, themes_regex=PUZZLE_EVAL_THEMES
        )
    else:
        tqdm.write(f"  WARNING: {PUZZLE_CSV} non trovato, probe puzzle disabilitato")

    # Istanza MCTS
    main_mcts = BatchedPointerMCTS(main_model, DEVICE, c_puct=C_PUCT, leaves_per_tree=8)

    print(f"Parametri: {sum(p.numel() for p in main_model.parameters()):,}")
    print(f"Inizio AlphaZero training per {EPOCHS} epoche (da epoca {start_epoch})\n")

    epoch_bar = tqdm(range(start_epoch, EPOCHS + 1), desc="Epoche AZ", dynamic_ncols=True)

    for epoch in epoch_bar:

        # ----------------------------------------------------------------
        # Self-play batched con curriculum
        # ----------------------------------------------------------------
        wins = draws = losses = 0
        new_steps = 0

        start_fens = []
        for i in range(GAMES_PER_EPOCH):
            if curriculum_ds and random.random() < CURRICULUM_PROB:
                start_fens.append(curriculum_ds.get_random_fen())
            else:
                start_fens.append(None)

        n_curriculum_games = sum(f is not None for f in start_fens)
        tqdm.write(f"  Epoch {epoch:03d} [self-play batched {GAMES_PER_EPOCH} partite, "
                   f"{n_curriculum_games} curriculum]...")

        all_steps, terminals = main_mcts.play_games_batched(
            n_games         = GAMES_PER_EPOCH,
            num_simulations = NUM_SIMULATIONS,
            temp_high       = TEMP_HIGH,
            temp_low        = TEMP_LOW,
            temp_threshold  = TEMP_THRESHOLD,
            max_moves       = MAX_MOVES,
            start_fens      = start_fens,
        )

        for steps, terminal in zip(all_steps, terminals):
            replay_buffer.extend(steps)
            new_steps += len(steps)
            if terminal > 0:   wins   += 1
            elif terminal < 0: losses += 1
            else:              draws  += 1

        tqdm.write(f"  Self-play: {new_steps} nuovi step, "
                   f"W{wins}/D{draws}/L{losses}, buffer={len(replay_buffer)}")

        # ----------------------------------------------------------------
        # Training sul buffer
        # ----------------------------------------------------------------
        if len(replay_buffer) >= MIN_BUFFER:
            loss_stats = train_on_buffer(
                main_model, optimizer, list(replay_buffer),
                TRAIN_STEPS, DEVICE, mixed_ds
            )
            p_loss_str = f"{loss_stats['policy_loss']:.4f}"
            v_loss_str = f"{loss_stats['value_loss']:.4f}"
            scheduler.step(loss_stats["policy_loss"] + loss_stats["value_loss"])
        else:
            p_loss_str = "N/A (buffer piccolo)"
            v_loss_str = "N/A"
            tqdm.write(f"  Buffer troppo piccolo ({len(replay_buffer)}/{MIN_BUFFER}), salto training.")

        # ----------------------------------------------------------------
        # Probe su puzzle Lichess (held-out, ogni PUZZLE_EVAL_EVERY epoche)
        # ----------------------------------------------------------------
        score_updated = False
        if puzzle_eval is not None and (epoch % PUZZLE_EVAL_EVERY == 0 or epoch == start_epoch):
            tqdm.write("  Probe su puzzle Lichess (held-out)...")
            stats = puzzle_eval.evaluate(
                main_model, main_mcts,
                mcts_n_samples=PUZZLE_MCTS_SAMPLES,
                mcts_num_simulations=PUZZLE_MCTS_SIMS,
            )
            score = puzzle_score(stats)
            tqdm.write(
                f"  Puzzle probe (n={stats['n']}): "
                f"value_calib={stats['value_calibration_rate']:.3f}  "
                f"policy_solve={stats['policy_solve_rate']:.3f}  "
                f"mcts_solve={stats.get('mcts_solve_rate', float('nan')):.3f}  "
                f"value_mean={stats['value_mean']:.3f}  "
                f"score={score:.3f}"
            )

            if score > best_score:
                best_score = score
                score_updated = True
                tqdm.write(f"  ★ Nuovo best score: {best_score:.3f}")

        # ----------------------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------------------
        checkpoint_state = {
            "epoch":      epoch,
            "model":      main_model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "best_score": best_score,
        }

        save_checkpoint(checkpoint_state, az_last, replay_buffer)

        if score_updated:
            save_checkpoint(checkpoint_state, az_best, replay_buffer)

        tqdm.write(
            f"Epoch {epoch:03d} | "
            f"Self-play W/D/L: {wins}/{draws}/{losses}  "
            f"buf: {len(replay_buffer)}  "
            f"new_steps: {new_steps}  "
            f"p_loss: {p_loss_str}  "
            f"v_loss: {v_loss_str}  "
            f"best_score: {best_score:.3f}  "
            f"LR: {optimizer.param_groups[1]['lr']:.2e}"
        )

        epoch_bar.set_postfix({
            "p_loss": p_loss_str,
            "v_loss": v_loss_str,
            "best":   f"{best_score:.3f}",
            "buf":    len(replay_buffer),
        })

    print(f"\nTraining completato. Best score: {best_score:.3f}")


if __name__ == "__main__":
    main()
