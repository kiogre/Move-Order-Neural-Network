"""
train_alphazero.py — Training AlphaZero-style per JellyFishPointer.

Pipeline per ogni epoca:
  1. Self-play con MCTS: genera partite usando get_policy_target per ogni mossa
  2. Accumula nel replay buffer: (board, policy_target, value_target)
  3. Training supervisionato sui target MCTS

Loss:
  policy loss : cross-entropy tra policy_target MCTS e output della rete
  value  loss : MSE tra valore predetto e risultato della partita
"""

import os
import copy
import random
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import deque
from tqdm import tqdm

# Adatta questi import ai tuoi nomi file
from MLChess import encode_board, encode_legal_moves, JellyFishPointer, PointerMCTS

MOVE_VECTOR_DIM = 46

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

SUPERVISED_CHECKPOINT = "checkpoints_pointer/best.pt"
AZ_CHECKPOINT_DIR     = "checkpoints_az"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS            = 500
GAMES_PER_EPOCH   = 40 #32            # partite self-play per epoca (MCTS è lento)
MAX_MOVES         = 300           # tetto mosse per partita

# MCTS
NUM_SIMULATIONS   = 100           # simulazioni per mossa
TEMP_HIGH         = 1.0           # temperatura prime TEMP_THRESHOLD mosse
TEMP_LOW          = 0.1           # temperatura mosse successive (più greedy)
TEMP_THRESHOLD    = 30            # dopo N mosse passa a temperatura bassa

# Replay buffer
BUFFER_SIZE       = 50_000        # massimo step nel buffer
MIN_BUFFER        = 1_000         # minimo step prima di iniziare il training

# Training
TRAIN_STEPS       = 200           # step di gradient update per epoca
BATCH_SIZE        = 256
LR_BACKBONE       = 1e-5
LR_HEADS          = 1e-4

# Frozen opponent
EVAL_GAMES        = 20
WINRATE_THRESHOLD = 0.52 #0.55


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
# Self-play con MCTS — genera step per il replay buffer
# ---------------------------------------------------------------------------

def play_game_mcts(
    mcts:          PointerMCTS,
    frozen_mcts:   PointerMCTS,
    main_is_white: bool = True,
) -> list[dict]:
    """
    Gioca una partita completa usando MCTS per entrambi i lati.
    Raccoglie (board_fen, policy_target, legal_moves_tensor) per ogni mossa.
    Il value_target viene assegnato alla fine con il risultato della partita.

    Returns:
        steps : lista di dict con board_fen, policy_target, legal_moves, value_target
    """
    board   = chess.Board()
    steps   = []   # (board_fen, policy_target, legal_moves_tensor)
    move_num = 0

    while not board.is_game_over() and move_num < MAX_MOVES:
        is_main = (board.turn == chess.WHITE) == main_is_white
        active_mcts = mcts if is_main else frozen_mcts

        # Temperatura: alta nelle prime mosse, bassa dopo
        temp = TEMP_HIGH if move_num < TEMP_THRESHOLD else TEMP_LOW

        # Policy target dalla distribuzione delle visite MCTS
        policy_target = active_mcts.get_policy_target(
            board,
            num_simulations=NUM_SIMULATIONS,
            temperature=temp,
        )   # {chess.Move: float}

        # Salva lo step solo per le mosse del main model
        if is_main:
            legal_moves_list = list(board.legal_moves)
            legal_moves_t    = encode_legal_moves(board)   # (n, 46)

            # Vettore target allineato all'ordine di legal_moves_list
            target_vec = torch.zeros(len(legal_moves_list))
            for j, move in enumerate(legal_moves_list):
                target_vec[j] = policy_target.get(move, 0.0)
            # Rinormalizza per sicurezza
            s = target_vec.sum()
            if s > 0:
                target_vec = target_vec / s

            steps.append({
                "board_fen":     board.fen(),
                "legal_moves":   legal_moves_t,   # (n, 46)
                "policy_target": target_vec,       # (n,)
                "value_target":  None,             # riempito dopo
            })

        # Scegli la mossa dalla distribuzione MCTS
        moves  = list(policy_target.keys())
        probs  = torch.tensor([policy_target[m] for m in moves], dtype=torch.float32)
        idx    = torch.multinomial(probs, 1).item()
        move   = moves[idx]
        board.push(move)
        move_num += 1

    # Assegna value_target in base al risultato
    result = board.result()
    if result == "1-0":
        terminal = 1.0 if main_is_white else -1.0
    elif result == "0-1":
        terminal = -1.0 if main_is_white else 1.0
    else:
        terminal = 0.0

    # Dal punto di vista del giocatore che ha mosso: alterna segno
    # (gli step sono tutti del main model, che gioca un colore fisso)
    for step in steps:
        step["value_target"] = terminal

    return steps, terminal


# ---------------------------------------------------------------------------
# Valutazione winrate contro frozen (greedy MCTS, temp bassa)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_vs_frozen(
    main_mcts:   PointerMCTS,
    frozen_mcts: PointerMCTS,
    n_games:     int = EVAL_GAMES,
) -> tuple[float, int, int, int]:
    wins = draws = losses = 0

    for i in range(n_games):
        main_white = (i % 2 == 0)
        board      = chess.Board()
        move_num   = 0

        while not board.is_game_over() and move_num < MAX_MOVES:
            is_main    = (board.turn == chess.WHITE) == main_white
            active     = main_mcts if is_main else frozen_mcts
            move       = active.get_best_move(board, num_simulations=NUM_SIMULATIONS, temperature=0.0)
            board.push(move)
            move_num  += 1

        result = board.result()
        if result == "1-0":
            if main_white: wins += 1
            else:          losses += 1
        elif result == "0-1":
            if main_white: losses += 1
            else:          wins += 1
        else:
            draws += 1

    winrate = (wins + 0.5 * draws) / n_games
    return winrate, wins, draws, losses


# ---------------------------------------------------------------------------
# Training step sui dati del replay buffer
# ---------------------------------------------------------------------------

def train_on_buffer(
    model:     JellyFishPointer,
    optimizer: Adam,
    buffer:    list[dict],
    n_steps:   int,
    device:    torch.device,
) -> dict:
    model.train()

    total_policy_loss = 0.0
    total_value_loss  = 0.0

    for _ in range(n_steps):
        batch = random.sample(buffer, min(BATCH_SIZE, len(buffer)))

        board_tensors  = []
        moves_tensors  = []
        policy_targets = []
        value_targets  = []

        for step in batch:
            board_tensors.append(encode_board(step["board_fen"]))
            moves_tensors.append(step["legal_moves"])
            policy_targets.append(step["policy_target"])
            value_targets.append(step["value_target"])

        # Padding mosse
        max_n = max(m.shape[0] for m in moves_tensors)
        B     = len(batch)

        moves_padded    = torch.zeros(B, max_n, MOVE_VECTOR_DIM, device=device)
        move_mask       = torch.zeros(B, max_n, dtype=torch.bool, device=device)
        policy_padded   = torch.zeros(B, max_n, device=device)

        for i, (m, p) in enumerate(zip(moves_tensors, policy_targets)):
            n = m.shape[0]
            moves_padded[i, :n]  = m.to(device)
            move_mask[i, :n]     = True
            policy_padded[i, :n] = p.to(device)

        boards_t  = torch.stack(board_tensors).to(device)
        values_t  = torch.tensor(value_targets, dtype=torch.float32, device=device).unsqueeze(1)

        # Forward
        logits, probs, value_pred = model(boards_t, moves_padded, move_mask)

        # Policy loss — KL divergence tra target MCTS e output rete
        # Equivalente a cross-entropy con distribuzione soft
        log_probs   = torch.log(probs + 1e-8)
        policy_loss = -(policy_padded * log_probs).sum(dim=1).mean()

        # Value loss
        value_loss  = F.mse_loss(value_pred, values_t)

        loss = policy_loss + value_loss

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

import pickle

def save_checkpoint(state: dict, path: str, replay_buffer):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buffer_path = path.replace(".pt", "_buffer.pkl")
    with open(buffer_path, "wb") as f:
        pickle.dump(list(replay_buffer), f)
    torch.save(state, path)
    tqdm.write(f"  → checkpoint salvato: {path}")


def load_checkpoint(path, model, optimizer, scheduler, replay_buffer):
    ckpt       = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt and scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    epoch      = ckpt.get("epoch", 0)
    best_wr    = ckpt.get("best_winrate", 0.0)
    frozen_sd  = ckpt.get("frozen_state_dict", None)
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

    main_model = torch.compile(main_model)
    frozen_model = torch.compile(frozen_model)

    optimizer = build_optimizer(main_model)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)

    start_epoch  = 1
    best_winrate = 0.0
    replay_buffer: deque[dict] = deque(maxlen=BUFFER_SIZE)

    az_last = os.path.join(AZ_CHECKPOINT_DIR, "last.pt")

    if os.path.exists(az_last):
        print("Checkpoint AlphaZero trovato, riprendo...")
        start_epoch, best_winrate, frozen_sd = load_checkpoint(
            az_last, main_model, optimizer, scheduler, replay_buffer
        )
        start_epoch += 1
        if frozen_sd is not None:
            frozen_model.load_state_dict(frozen_sd)
        else:
            frozen_model.load_state_dict(main_model.state_dict())

    elif os.path.exists(SUPERVISED_CHECKPOINT):
        print(f"Carico supervised: {SUPERVISED_CHECKPOINT}")
        ckpt = torch.load(SUPERVISED_CHECKPOINT, map_location=DEVICE)
        main_model.load_state_dict(ckpt["model"])
        frozen_model.load_state_dict(ckpt["model"])

    else:
        print("Nessun checkpoint trovato, parto da zero.")
        frozen_model.load_state_dict(main_model.state_dict())

    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad = False

    main_model.eval()

    # Crea istanze MCTS
    main_mcts   = PointerMCTS(main_model,   DEVICE)
    frozen_mcts = PointerMCTS(frozen_model, DEVICE)

    print(f"Parametri: {sum(p.numel() for p in main_model.parameters()):,}")
    print(f"Inizio AlphaZero training per {EPOCHS} epoche (da epoca {start_epoch})\n")

    epoch_bar = tqdm(range(start_epoch, EPOCHS + 1), desc="Epoche AZ", dynamic_ncols=True)

    for epoch in epoch_bar:

        # ----------------------------------------------------------------
        # Self-play con MCTS
        # ----------------------------------------------------------------
        wins = draws = losses = 0
        new_steps = 0

        game_bar = tqdm(
            range(GAMES_PER_EPOCH),
            desc=f"  Epoch {epoch:03d} [self-play]",
            leave=False,
            dynamic_ncols=True,
        )

        for game_idx in game_bar:
            main_white = (game_idx % 2 == 0)
            steps, terminal = play_game_mcts(main_mcts, frozen_mcts, main_is_white=main_white)

            replay_buffer.extend(steps)
            new_steps += len(steps)

            if terminal > 0:   wins   += 1
            elif terminal < 0: losses += 1
            else:              draws  += 1

            game_bar.set_postfix({
                "W": wins, "D": draws, "L": losses,
                "buf": len(replay_buffer),
            })

        # ----------------------------------------------------------------
        # Training sul buffer
        # ----------------------------------------------------------------
        if len(replay_buffer) >= MIN_BUFFER:
            loss_stats = train_on_buffer(main_model, optimizer, list(replay_buffer), TRAIN_STEPS, DEVICE)
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
        winrate, w, d, l = evaluate_vs_frozen(main_mcts, frozen_mcts)
        tqdm.write(f"  Winrate: {winrate:.3f}  (W{w}/D{d}/L{l})")

        frozen_updated = False
        if winrate >= WINRATE_THRESHOLD:
            frozen_model.load_state_dict(main_model.state_dict())
            frozen_model.eval()
            for p in frozen_model.parameters():
                p.requires_grad = False
            frozen_mcts = PointerMCTS(frozen_model, DEVICE)
            frozen_updated = True
            tqdm.write(f"  ★ Frozen aggiornato! Winrate: {winrate:.3f}")

        scheduler.step(winrate)

        tqdm.write(
            f"Epoch {epoch:03d} | "
            f"Self-play W/D/L: {wins}/{draws}/{losses}  "
            f"buf: {len(replay_buffer)}  "
            f"new_steps: {new_steps}  "
            f"winrate_vs_frozen: {winrate:.3f}{'  [frozen aggiornato]' if frozen_updated else ''}  "
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

        save_checkpoint(checkpoint_state, os.path.join(AZ_CHECKPOINT_DIR, "last.pt"), replay_buffer)

        if winrate > best_winrate:
            best_winrate = winrate
            checkpoint_state["best_winrate"] = best_winrate
            save_checkpoint(checkpoint_state, os.path.join(AZ_CHECKPOINT_DIR, "best.pt"), replay_buffer)
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
