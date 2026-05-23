import os
import copy
import random
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Adatta questi import ai tuoi nomi file
from MLChess import encode_board, encode_legal_moves, JellyFishPointer

MOVE_VECTOR_DIM = 46

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

SUPERVISED_CHECKPOINT = "checkpoints_value/best.pt"
RL_CHECKPOINT_DIR     = "checkpoints_rl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS            = 500
GAMES_PER_EPOCH   = 128       # partite di training per epoca
MAX_MOVES         = 300       # tetto mosse per evitare partite infinite

# Valutazione avversario
EVAL_GAMES        = 40        # partite per valutare il winrate contro frozen
WINRATE_THRESHOLD = 0.52 #0.55      # soglia per aggiornare la copia frozen

# PPO
PPO_EPOCHS        = 4
PPO_CLIP          = 0.1
PPO_BATCH_SIZE    = 256
GAMMA             = 0.99
LAMBDA_GAE        = 0.95
ENTROPY_COEF      = 0.15
VALUE_COEF        = 0.5

# Temperatura: parte alta per esplorare, scende verso 1.0
TEMP_START        = 2.0
TEMP_END          = 1.0
TEMP_DECAY_EPOCHS = 200       # epoche per scendere da TEMP_START a TEMP_END

# Learning rates
LR_BACKBONE       = 1e-5
LR_HEADS          = 1e-4


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_temperature(epoch: int) -> float:
    """Temperatura lineare decrescente da TEMP_START a TEMP_END."""
    t = min(epoch / TEMP_DECAY_EPOCHS, 1.0)
    return TEMP_START + t * (TEMP_END - TEMP_START)


def build_optimizer(model: JellyFishPointer) -> Adam:
    return Adam([
        {"params": list(model.backbone.parameters()) +
                   list(model.move_encoder.parameters()),
         "lr": LR_BACKBONE,
         "name": "backbone"},
        {"params": list(model.policy_head.parameters()) +
                   list(model.value_head.parameters()),
         "lr": LR_HEADS,
         "name": "heads"},
    ])


# ---------------------------------------------------------------------------
# Inferenza
# ---------------------------------------------------------------------------

@torch.no_grad()
def select_move(
    model: JellyFishPointer,
    board: chess.Board,
    device: torch.device,
    temperature: float = 1.0,
    greedy: bool = False,
) -> tuple:
    """
    Seleziona una mossa dalla policy della rete.

    Returns:
        move, move_idx, log_prob, value
    """
    legal_moves_list = list(board.legal_moves)
    if not legal_moves_list:
        return None, -1, 0.0, 0.0

    board_tensor = encode_board(board.fen()).unsqueeze(0).to(device)
    moves_tensor = encode_legal_moves(board).unsqueeze(0).to(device)

    logits, probs, value = model(board_tensor, moves_tensor)

    if temperature != 1.0:
        logits = logits / temperature
        probs  = torch.softmax(logits, dim=-1)

    if greedy:
        move_idx = probs[0].argmax().item()
        log_prob = torch.log(probs[0, move_idx] + 1e-8).item()
    else:
        dist     = torch.distributions.Categorical(probs=probs[0])
        move_idx = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(move_idx, device=device)).item()

    return legal_moves_list[move_idx], move_idx, log_prob, value[0, 0].item()


# ---------------------------------------------------------------------------
# Generazione partita — self-play
# ---------------------------------------------------------------------------

def play_game_selfplay(
    main_model:   JellyFishPointer,
    frozen_model: JellyFishPointer,
    device:       torch.device,
    temperature:  float = 1.0,
    main_is_white: bool = True,
) -> tuple[list[dict], float]:
    """
    Gioca una partita tra main_model e frozen_model.

    Returns:
        trajectory      : step del main_model con campi per PPO
        terminal_reward : +1 vittoria, -1 sconfitta, 0 patta (dal punto di vista di main)
    """
    main_model.eval()
    frozen_model.eval()

    board      = chess.Board()
    trajectory = []

    for _ in range(MAX_MOVES):
        if board.is_game_over():
            break

        is_main_turn = (board.turn == chess.WHITE) == main_is_white

        if is_main_turn:
            pre_move_fen = board.fen()
            move, move_idx, log_prob, value = select_move(
                main_model, board, device, temperature=temperature
            )
            if move is None:
                break
            board.push(move)
            trajectory.append({
                "pre_move_fen": pre_move_fen,
                "move_idx":     move_idx,
                "log_prob_old": log_prob,
                "value":        value,
                "done":         board.is_game_over(),
            })
        else:
            move, _, _, _ = select_move(
                frozen_model, board, device, temperature=1.0
            )
            if move is None:
                break
            board.push(move)

    # Reward terminale
    result = board.result()
    if result == "1-0":
        terminal_reward = 1.0 if main_is_white else -1.0
    elif result == "0-1":
        terminal_reward = -1.0 if main_is_white else 1.0
    else:
        terminal_reward = 0.0

    # Assegna reward solo all'ultimo step
    if trajectory:
        trajectory[-1]["reward"] = terminal_reward
        for step in trajectory[:-1]:
            step["reward"] = 0.0

    return trajectory, terminal_reward


# ---------------------------------------------------------------------------
# Valutazione winrate contro frozen
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_vs_frozen(
    main_model:   JellyFishPointer,
    frozen_model: JellyFishPointer,
    device:       torch.device,
    n_games:      int = EVAL_GAMES,
) -> float:
    """
    Gioca n_games partite greedy (no esplorazione) e restituisce il winrate.
    """
    main_model.eval()
    frozen_model.eval()

    wins = draws = losses = 0

    for i in range(n_games):
        main_white = (i % 2 == 0)
        board      = chess.Board()

        for _ in range(MAX_MOVES):
            if board.is_game_over():
                break
            is_main = (board.turn == chess.WHITE) == main_white
            model   = main_model if is_main else frozen_model
            move, _, _, _ = select_move(model, board, device, temperature=1.2)
            if move is None:
                break
            board.push(move)

        result = board.result()
        if result == "1-0":
            if main_white:
                wins += 1
            else:
                losses += 1
        elif result == "0-1":
            if main_white:
                losses += 1
            else:
                wins += 1
        else:
            draws += 1

    winrate = (wins + 0.5 * draws) / n_games
    return winrate, wins, draws, losses


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(trajectory: list[dict]) -> list[dict]:
    rewards    = [s["reward"] for s in trajectory]
    values     = [s["value"]  for s in trajectory]
    dones      = [s["done"]   for s in trajectory]

    advantages = []
    gae        = 0.0
    next_value = 0.0

    for t in reversed(range(len(trajectory))):
        delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t]
        gae   = delta + GAMMA * LAMBDA_GAE * (1 - dones[t]) * gae
        advantages.insert(0, gae)
        next_value = values[t]

    returns = [adv + val for adv, val in zip(advantages, values)]

    for i, step in enumerate(trajectory):
        step["advantage"] = advantages[i]
        step["return"]    = returns[i]

    return trajectory


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    model:     JellyFishPointer,
    optimizer: Adam,
    buffer:    list[dict],
    device:    torch.device,
) -> dict:
    advantages = torch.tensor([s["advantage"] for s in buffer], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_entropy     = 0.0
    n_updates         = 0

    model.train()

    for _ in range(PPO_EPOCHS):
        indices = list(range(len(buffer)))
        random.shuffle(indices)

        for start in range(0, len(buffer), PPO_BATCH_SIZE):
            batch_idx = indices[start:start + PPO_BATCH_SIZE]
            if not batch_idx:
                continue

            board_tensors = []
            moves_tensors = []
            move_indices  = []
            old_log_probs = []
            returns_      = []
            advs          = []

            for idx in batch_idx:
                step  = buffer[idx]
                board = chess.Board(step["pre_move_fen"])
                board_tensors.append(encode_board(step["pre_move_fen"]))
                moves_tensors.append(encode_legal_moves(board))
                move_indices.append(step["move_idx"])
                old_log_probs.append(step["log_prob_old"])
                returns_.append(step["return"])
                advs.append(advantages[idx].item())

            max_n = max(m.shape[0] for m in moves_tensors)
            B     = len(batch_idx)

            moves_padded = torch.zeros(B, max_n, MOVE_VECTOR_DIM, device=device)
            move_mask    = torch.zeros(B, max_n, dtype=torch.bool, device=device)

            for i, m in enumerate(moves_tensors):
                n = m.shape[0]
                moves_padded[i, :n] = m.to(device)
                move_mask[i, :n]    = True

            board_t   = torch.stack(board_tensors).to(device)
            labels_t  = torch.tensor(move_indices,  dtype=torch.long,    device=device)
            old_lp_t  = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
            returns_t = torch.tensor(returns_,      dtype=torch.float32, device=device).unsqueeze(1)
            adv_t     = torch.tensor(advs,          dtype=torch.float32, device=device)

            logits, probs, value_pred = model(board_t, moves_padded, move_mask)

            dist          = torch.distributions.Categorical(probs=probs)
            new_log_probs = dist.log_prob(labels_t)
            entropy       = dist.entropy().mean()

            ratio  = torch.exp(new_log_probs - old_lp_t)
            surr1  = ratio * adv_t
            surr2  = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv_t
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(value_pred, returns_t)
            loss   = p_loss + VALUE_COEF * v_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_policy_loss += p_loss.item()
            total_value_loss  += v_loss.item()
            total_entropy     += entropy.item()
            n_updates         += 1

    model.eval()

    return {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "value_loss":  total_value_loss  / max(n_updates, 1),
        "entropy":     total_entropy     / max(n_updates, 1),
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    tqdm.write(f"  → checkpoint salvato: {path}")


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt        = torch.load(path, map_location=DEVICE)
    state_dict = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt and scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    epoch       = ckpt.get("epoch", 0)
    best_wr     = ckpt.get("best_winrate", 0.0)
    frozen_sd   = ckpt.get("frozen_state_dict", None)
    tqdm.write(f"  → checkpoint caricato: {path}  (epoch {epoch}, best winrate {best_wr:.3f})")
    return epoch, best_wr, frozen_sd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    main_model   = JellyFishPointer().to(DEVICE)
    frozen_model = JellyFishPointer().to(DEVICE)

    optimizer = build_optimizer(main_model)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)

    start_epoch  = 1
    best_winrate = 0.0

    rl_last = os.path.join(RL_CHECKPOINT_DIR, "last.pt")

    if os.path.exists(rl_last):
        print("Checkpoint RL trovato, riprendo...")
        start_epoch, best_winrate, frozen_sd = load_checkpoint(
            rl_last, main_model, optimizer, scheduler
        )
        start_epoch += 1
        if frozen_sd is not None:
            if any(k.startswith("_orig_mod.") for k in frozen_sd.keys()):
                frozen_sd = {k.replace("_orig_mod.", ""): v for k, v in frozen_sd.items()}
            frozen_model.load_state_dict(frozen_sd)
        else:
            frozen_model.load_state_dict(main_model.state_dict())

    elif os.path.exists(SUPERVISED_CHECKPOINT):
        print(f"Carico supervised: {SUPERVISED_CHECKPOINT}")
        ckpt = torch.load(SUPERVISED_CHECKPOINT, map_location=DEVICE)
        state_dict = ckpt["model"]
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        main_model.load_state_dict(state_dict)
        frozen_model.load_state_dict(state_dict)  # frozen parte uguale a main

    else:
        print("Nessun checkpoint trovato, parto da zero.")
        frozen_model.load_state_dict(main_model.state_dict())

    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad = False

    print(f"Parametri: {sum(p.numel() for p in main_model.parameters()):,}")
    print(f"Inizio self-play RL per {EPOCHS} epoche (da epoca {start_epoch})\n")

    epoch_bar = tqdm(range(start_epoch, EPOCHS + 1), desc="Epoche RL", dynamic_ncols=True)

    for epoch in epoch_bar:
        temperature = get_temperature(epoch)

        # --------------------------------------------------------------------
        # Generazione partite
        # --------------------------------------------------------------------
        buffer       = []
        game_rewards = []
        wins = draws = losses = 0

        game_bar = tqdm(
            range(GAMES_PER_EPOCH),
            desc=f"  Epoch {epoch:03d} [gioco]",
            leave=False,
            dynamic_ncols=True,
        )

        for game_idx in game_bar:
            main_white = (game_idx % 2 == 0)
            trajectory, terminal_reward = play_game_selfplay(
                main_model, frozen_model, DEVICE,
                temperature=temperature,
                main_is_white=main_white,
            )

            if not trajectory:
                continue

            trajectory = compute_gae(trajectory)
            buffer.extend(trajectory)
            game_rewards.append(terminal_reward)

            if terminal_reward > 0:
                wins += 1
            elif terminal_reward == 0.0:
                draws += 1
            else:
                losses += 1

            game_bar.set_postfix({
                "W": wins, "D": draws, "L": losses,
                "temp": f"{temperature:.2f}",
                "buf":  len(buffer),
            })

        if not buffer:
            tqdm.write(f"Epoch {epoch:03d} | Buffer vuoto, salto.")
            continue

        avg_reward = sum(game_rewards) / len(game_rewards)

        # --------------------------------------------------------------------
        # PPO update
        # --------------------------------------------------------------------
        loss_stats = ppo_update(main_model, optimizer, buffer, DEVICE)

        # --------------------------------------------------------------------
        # Valutazione winrate contro frozen
        # --------------------------------------------------------------------
        tqdm.write(f"  Valutazione contro frozen ({EVAL_GAMES} partite)...")
        winrate, w, d, l = evaluate_vs_frozen(main_model, frozen_model, DEVICE)
        tqdm.write(f"  Winrate: {winrate:.3f}  (W{w}/D{d}/L{l})")

        # Aggiorna frozen se supera la soglia
        frozen_updated = False
        if winrate >= WINRATE_THRESHOLD:
            frozen_model.load_state_dict(main_model.state_dict())
            frozen_model.eval()
            for p in frozen_model.parameters():
                p.requires_grad = False
            frozen_updated = True
            tqdm.write(f"  ★ Frozen aggiornato! Winrate: {winrate:.3f}")

        scheduler.step(winrate)

        tqdm.write(
            f"Epoch {epoch:03d} | "
            f"Train W/D/L: {wins}/{draws}/{losses}  "
            f"avg_reward: {avg_reward:+.3f}  "
            f"winrate_vs_frozen: {winrate:.3f}{'  [frozen aggiornato]' if frozen_updated else ''}  "
            f"p_loss: {loss_stats['policy_loss']:.4f}  "
            f"v_loss: {loss_stats['value_loss']:.4f}  "
            f"entropy: {loss_stats['entropy']:.4f}  "
            f"temp: {temperature:.2f}  "
            f"LR: {optimizer.param_groups[1]['lr']:.2e}"
        )

        checkpoint_state = {
            "epoch":              epoch,
            "model":              main_model.state_dict(),
            "frozen_state_dict":  frozen_model.state_dict(),
            "optimizer":          optimizer.state_dict(),
            "scheduler":          scheduler.state_dict(),
            "best_winrate":       best_winrate,
        }

        save_checkpoint(checkpoint_state, os.path.join(RL_CHECKPOINT_DIR, "last.pt"))

        if winrate > best_winrate:
            best_winrate = winrate
            checkpoint_state["best_winrate"] = best_winrate
            save_checkpoint(checkpoint_state, os.path.join(RL_CHECKPOINT_DIR, "best.pt"))
            tqdm.write(f"  ★ Nuovo best winrate: {best_winrate:.3f}")

        epoch_bar.set_postfix({
            "winrate":  f"{winrate:.3f}",
            "best":     f"{best_winrate:.3f}",
            "temp":     f"{temperature:.2f}",
            "frozen":   "✓" if frozen_updated else "-",
        })

    print(f"\nTraining completato. Best winrate: {best_winrate:.3f}")


if __name__ == "__main__":
    main()
