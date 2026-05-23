"""
train_alphazero_v2.py — Training AlphaZero-style per JellyFishPointer (BatchedPointerMCTS).

Pipeline per ogni epoca:
  1. Self-play batched con MCTS: genera N partite in parallelo
  2. Accumula nel replay buffer: (board, policy_target, value_target)
  3. Training supervisionato sui target MCTS + campioni curriculum

Differenze rispetto a train_alphazero.py (v1 sequenziale):
  - BatchedPointerMCTS: tutte le partite in parallelo, una forward pass per step
  - Frozen greedy gestito internamente da play_games_batched
  - Curriculum learning: 30% delle partite partono da posizioni tattiche
  - Mixed buffer: 20% del batch da dataset tattico (solo policy loss)
  - Value loss weight: 3.0 per ricalibrazione del value head
  - Checkpoint atomico con os.replace()
  - best_winrate letto da best.pt separatamente
"""

import os
import copy
import math
import random
import pickle
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import deque
from tqdm import tqdm

from MLChess import encode_board, encode_legal_moves, JellyFishPointer, BatchedPointerMCTS

MOVE_VECTOR_DIM = 46

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

SUPERVISED_CHECKPOINT = "checkpoints_rl/last.pt"
AZ_CHECKPOINT_DIR     = "checkpoints_az_v2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS          = 500
GAMES_PER_EPOCH = 64
MAX_MOVES       = 400

# MCTS
NUM_SIMULATIONS = 150
TEMP_HIGH       = 1.0
TEMP_LOW        = 0.01
TEMP_THRESHOLD  = 10

# Replay buffer
BUFFER_SIZE = 200_000
MIN_BUFFER  = 1_000

# Training
TRAIN_STEPS      = 200
BATCH_SIZE       = 256
LR_BACKBONE      = 1e-5
LR_HEADS         = 1e-4
VALUE_LOSS_WEIGHT = 3.0   # peso value loss per ricalibrazione scala predizioni

# Frozen opponent
EVAL_GAMES        = 100
WINRATE_THRESHOLD = 0.55

# Curriculum learning
CURRICULUM_CSV       = "../over_mate_1_tactic_evals.csv"
CURRICULUM_PROB      = 0.30
CURRICULUM_MAX_MOVES = 120

# Mixed buffer — campioni diretti dal dataset tattico
MIXED_BUFFER_RATIO = 0.0
MIXED_BUFFER_SIZE  = 50_000


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
# Curriculum dataset
# ---------------------------------------------------------------------------

def encode_value(evaluation: str, max_cp: int = 1000) -> float:
    if '#' in str(evaluation):
        return 1.0 if '+' in str(evaluation) or not str(evaluation).startswith('-') else -1.0
    try:
        cp = max(-max_cp, min(max_cp, int(evaluation)))
        return cp / max_cp
    except (ValueError, TypeError):
        return 0.0


class CurriculumDataset:
    """
    Carica posizioni dal dataset tattico per:
    1. Fornire FEN di partenza per le partite curriculum
    2. Fornire campioni diretti per il replay buffer misto (solo policy loss)
    """
    def __init__(self, csv_file: str, max_samples: int = MIXED_BUFFER_SIZE):
        tqdm.write(f"  Caricamento curriculum dataset da {csv_file}...")
        df = pd.read_csv(csv_file).dropna(subset=["FEN", "Evaluation", "Move"])
        if len(df) > max_samples * 4:
            df = df.sample(n=max_samples * 4, random_state=42)
        self.df       = df.reset_index(drop=True)
        self.max_size = max_samples
        tqdm.write(f"  Curriculum dataset: {len(self.df)} posizioni")

    def get_random_fen(self) -> str:
        return self.df.iloc[random.randint(0, len(self.df) - 1)]["FEN"]

    def get_mixed_samples(self, n: int) -> list[dict]:
        """
        Restituisce n campioni dal dataset.
        Policy target: one-hot sulla mossa Stockfish.
        Value target: None — escluso dalla value loss (centipawn non calibrati).
        """
        indices = random.sample(range(len(self.df)), min(n, len(self.df)))
        samples = []
        for idx in indices:
            row = self.df.iloc[idx]
            try:
                board      = chess.Board(row["FEN"])
                legal_list = list(board.legal_moves)
                if not legal_list:
                    continue

                target_move   = chess.Move.from_uci(str(row["Move"]))
                legal_moves_t = encode_legal_moves(board)

                target_vec = torch.zeros(len(legal_list))
                if target_move in legal_list:
                    target_vec[legal_list.index(target_move)] = 1.0
                else:
                    target_vec[0] = 1.0

                samples.append({
                    "board_fen":     row["FEN"],
                    "legal_moves":   legal_moves_t,
                    "policy_target": target_vec,
                    "value_target":  None,   # mascherato nella value loss
                })
            except Exception:
                continue
        return samples


# ---------------------------------------------------------------------------
# Valutazione winrate contro frozen
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_vs_frozen(
    main_mcts:   BatchedPointerMCTS,
    n_games:     int = EVAL_GAMES,
) -> tuple[float, int, int, int]:
    """
    Valutazione winrate usando play_games_batched — stessa infrastruttura del
    self-play, frozen greedy senza MCTS. Molto più veloce della versione
    sequenziale con get_best_move().

    main_is_white è alternato automaticamente dall'indice pari/dispari del game
    (i % 2 == 0 → main è bianco) — stessa logica della versione precedente.
    """
    wins = draws = losses = 0

    _, terminals = main_mcts.play_games_batched(
        n_games         = n_games,
        num_simulations = NUM_SIMULATIONS,
        temp_high       = 0.01,  # non zero: _visit_distribution fa 1/temp
        temp_low        = 0.01,
        temp_threshold  = 0,
        max_moves       = MAX_MOVES,
        start_fens      = None,
    )

    for terminal in terminals:
        if terminal > 0:   wins   += 1
        elif terminal < 0: losses += 1
        else:              draws  += 1

    winrate = (wins + 0.5 * draws) / n_games
    return winrate, wins, draws, losses


# ---------------------------------------------------------------------------
# Training sul replay buffer
# ---------------------------------------------------------------------------

def train_on_buffer(
    model:         JellyFishPointer,
    optimizer:     Adam,
    buffer:        list[dict],
    n_steps:       int,
    device:        torch.device,
    curriculum_ds: CurriculumDataset = None,
) -> dict:
    model.train()

    total_policy_loss = 0.0
    total_value_loss  = 0.0

    for _ in range(n_steps):
        n_curriculum = int(BATCH_SIZE * MIXED_BUFFER_RATIO) if curriculum_ds else 0
        n_buffer     = BATCH_SIZE - n_curriculum

        batch = random.sample(buffer, min(n_buffer, len(buffer)))

        if n_curriculum > 0 and curriculum_ds:
            mixed = curriculum_ds.get_mixed_samples(n_curriculum)
            batch = batch + mixed

        board_tensors  = []
        moves_tensors  = []
        policy_targets = []
        value_targets  = []

        for step in batch:
            board_tensors.append(encode_board(step["board_fen"]))
            moves_tensors.append(step["legal_moves"])
            policy_targets.append(step["policy_target"])
            value_targets.append(step["value_target"])  # None per campioni curriculum

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

        # Maschera value loss: escludi campioni curriculum (value_target = None)
        # I campioni buffer sono i primi n_buffer, i curriculum sono in coda
        value_mask = torch.tensor(
            [v is not None for v in value_targets],
            dtype=torch.bool, device=device
        )
        # Sostituisci None con 0.0 per costruire il tensore (mascherato dopo)
        values_clean = [v if v is not None else 0.0 for v in value_targets]
        values_t = torch.tensor(values_clean, dtype=torch.float32, device=device).unsqueeze(1)

        # Forward
        logits, probs, value_pred = model(boards_t, moves_padded, move_mask)

        # Policy loss — cross-entropy con distribuzione soft MCTS
        log_probs   = torch.log(probs + 1e-8)
        policy_loss = -(policy_padded * log_probs).sum(dim=1).mean()

        # Value loss — solo sui campioni MCTS (non curriculum)
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

    # Salvataggio atomico buffer
    buffer_path = path.replace(".pt", "_buffer.pkl")
    buffer_tmp  = buffer_path + ".tmp"
    with open(buffer_tmp, "wb") as f:
        pickle.dump(list(replay_buffer), f)
    os.replace(buffer_tmp, buffer_path)

    # Salvataggio atomico checkpoint
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
    best_wr   = ckpt.get("best_winrate", 0.0)
    frozen_sd = ckpt.get("frozen_state_dict", None)

    tqdm.write(f"  → checkpoint caricato: {path}  (epoch {epoch}, best winrate {best_wr:.3f})")

    buffer_path = path.replace(".pt", "_buffer.pkl")
    if os.path.exists(buffer_path):
        with open(buffer_path, "rb") as f:
            loaded = pickle.load(f)
        replay_buffer.extend(loaded)
        tqdm.write(f"  → buffer caricato: {len(replay_buffer)} step")

    return epoch, best_wr, frozen_sd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    main_model   = JellyFishPointer().to(DEVICE)
    frozen_model = JellyFishPointer().to(DEVICE)

    optimizer = build_optimizer(main_model)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=40)

    start_epoch  = 1
    best_winrate = 0.0
    replay_buffer: deque[dict] = deque(maxlen=BUFFER_SIZE)

    az_last = os.path.join(AZ_CHECKPOINT_DIR, "last.pt")
    az_best = os.path.join(AZ_CHECKPOINT_DIR, "best.pt")

    if os.path.exists(az_last):
        print("Checkpoint AlphaZero trovato, riprendo...")
        start_epoch, _, frozen_sd = load_checkpoint(
            az_last, main_model, optimizer, scheduler, replay_buffer
        )
        start_epoch += 1

        # Leggi best_winrate dal best.pt separatamente
        if os.path.exists(az_best):
            best_ckpt    = torch.load(az_best, map_location=DEVICE)
            best_winrate = best_ckpt.get("best_winrate", 0.0)
            tqdm.write(f"  → best winrate storico: {best_winrate:.3f}")

        if frozen_sd is not None:
            if any(k.startswith("_orig_mod.") for k in frozen_sd.keys()):
                frozen_sd = {k.replace("_orig_mod.", ""): v for k, v in frozen_sd.items()}
            frozen_model.load_state_dict(frozen_sd)
        else:
            frozen_model.load_state_dict(main_model.state_dict())

    elif os.path.exists(SUPERVISED_CHECKPOINT):
        print(f"Carico supervised: {SUPERVISED_CHECKPOINT}")
        ckpt = torch.load(SUPERVISED_CHECKPOINT, map_location=DEVICE)
        main_model.load_state_dict(ckpt["model"])
        frozen_model.load_state_dict(ckpt["model"])
        buffer_path = SUPERVISED_CHECKPOINT.replace(".pt", "_buffer.pkl")
        if os.path.exists(buffer_path):
            with open(buffer_path, "rb") as f:
                loaded = pickle.load(f)
            replay_buffer.extend(loaded)
            tqdm.write(f"  → buffer caricato: {len(replay_buffer)} step")

    else:
        print("Nessun checkpoint trovato, parto da zero.")
        frozen_model.load_state_dict(main_model.state_dict())

    # Ripristina LR esplicitamente (override del checkpoint)
    for group in optimizer.param_groups:
        if group["name"] == "backbone":
            group["lr"] = LR_BACKBONE
        else:
            group["lr"] = LR_HEADS

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=40)

    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad = False

    main_model.eval()

    # Curriculum dataset
    curriculum_ds = None
    if os.path.exists(CURRICULUM_CSV):
        curriculum_ds = CurriculumDataset(CURRICULUM_CSV)
    else:
        tqdm.write(f"  WARNING: {CURRICULUM_CSV} non trovato, curriculum disabilitato")

    # Istanze MCTS
    main_mcts   = BatchedPointerMCTS(main_model,   DEVICE, c_puct=2.5)
    frozen_mcts = BatchedPointerMCTS(frozen_model, DEVICE, c_puct=2.5)

    print(f"Parametri: {sum(p.numel() for p in main_model.parameters()):,}")
    print(f"Inizio AlphaZero training per {EPOCHS} epoche (da epoca {start_epoch})\n")

    epoch_bar = tqdm(range(start_epoch, EPOCHS + 1), desc="Epoche AZ", dynamic_ncols=True)

    for epoch in epoch_bar:

        # ----------------------------------------------------------------
        # Self-play batched con curriculum
        # ----------------------------------------------------------------
        wins = draws = losses = 0
        new_steps = 0

        # Prepara FEN di partenza: CURRICULUM_PROB% dal dataset tattico
        start_fens = []
        for i in range(GAMES_PER_EPOCH):
            if curriculum_ds and random.random() < CURRICULUM_PROB:
                start_fens.append(curriculum_ds.get_random_fen())
            else:
                start_fens.append(None)

        tqdm.write(f"  Epoch {epoch:03d} [self-play batched {GAMES_PER_EPOCH} partite, "
                   f"{sum(f is not None for f in start_fens)} curriculum]...")

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

        # ----------------------------------------------------------------
        # Training sul buffer
        # ----------------------------------------------------------------
        if len(replay_buffer) >= MIN_BUFFER:
            loss_stats = train_on_buffer(
                main_model, optimizer, list(replay_buffer),
                TRAIN_STEPS, DEVICE, curriculum_ds
            )
            p_loss_str = f"{loss_stats['policy_loss']:.4f}"
            v_loss_str = f"{loss_stats['value_loss']:.4f}"
        else:
            p_loss_str = "N/A (buffer piccolo)"
            v_loss_str = "N/A"
            tqdm.write(f"  Buffer troppo piccolo ({len(replay_buffer)}/{MIN_BUFFER}), salto training.")

        # ----------------------------------------------------------------
        # Valutazione winrate contro frozen
        # ----------------------------------------------------------------
        tqdm.write(f"  Valutazione contro frozen ({EVAL_GAMES} partite)...")
        winrate, w, d, l = evaluate_vs_frozen(main_mcts)
        tqdm.write(f"  Winrate: {winrate:.3f}  (W{w}/D{d}/L{l})")

        frozen_updated = False
        if winrate >= WINRATE_THRESHOLD:
            frozen_model.load_state_dict(main_model.state_dict())
            frozen_model.eval()
            for p in frozen_model.parameters():
                p.requires_grad = False
            frozen_mcts = BatchedPointerMCTS(frozen_model, DEVICE, c_puct=2.5)
            frozen_updated = True
            tqdm.write(f"  ★ Frozen aggiornato! Winrate: {winrate:.3f}")

        scheduler.step(winrate)

        tqdm.write(
            f"Epoch {epoch:03d} | "
            f"Self-play W/D/L: {wins}/{draws}/{losses}  "
            f"buf: {len(replay_buffer)}  "
            f"new_steps: {new_steps}  "
            f"winrate_vs_frozen: {winrate:.3f}"
            f"{'  [frozen aggiornato]' if frozen_updated else ''}  "
            f"p_loss: {p_loss_str}  "
            f"v_loss: {v_loss_str}  "
            f"LR: {optimizer.param_groups[1]['lr']:.2e}"
        )

        checkpoint_state = {
            "epoch":             epoch,
            "model":             main_model.state_dict(),
            "frozen_state_dict": frozen_model.state_dict(),
            "optimizer":         optimizer.state_dict(),
            "scheduler":         scheduler.state_dict(),
            "best_winrate":      best_winrate,
        }

        save_checkpoint(checkpoint_state, az_last, replay_buffer)

        if winrate > best_winrate:
            best_winrate = winrate
            checkpoint_state["best_winrate"] = best_winrate
            save_checkpoint(checkpoint_state, az_best, replay_buffer)
            tqdm.write(f"  ★ Nuovo best winrate: {best_winrate:.3f}")

        epoch_bar.set_postfix({
            "winrate": f"{winrate:.3f}",
            "best":    f"{best_winrate:.3f}",
            "buf":     len(replay_buffer),
            "frozen":  "✓" if frozen_updated else "-",
        })

    print(f"\nTraining completato. Best winrate: {best_winrate:.3f}")


if __name__ == "__main__":
    main()
