import os
import random
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import stockfish

# Adatta questi import ai tuoi nomi file
from MLChess import encode_board, encode_legal_moves
from MLChess import JellyFishPointer

MOVE_VECTOR_DIM = 46

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

SUPERVISED_CHECKPOINT = "checkpoints_pointer/best.pt"
RL_CHECKPOINT_DIR     = "checkpoints_rl"

STOCKFISH_PATH        = "/usr/bin/stockfish"
STOCKFISH_ELO         = 1320          # livello avversario fase 1

DEVICE                = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS                = 100
GAMES_PER_EPOCH       = 128 #64            # partite generate per epoca
MAX_MOVES_PER_GAME    = 200           # tetto per evitare partite infinite

# PPO
PPO_EPOCHS            = 2 #4             # quante volte ripassiamo il buffer per epoca
PPO_CLIP              = 0.1 #0.2
PPO_BATCH_SIZE        = 256           # step nel buffer per mini-batch PPO
GAMMA                 = 0.99          # discount factor
LAMBDA_GAE            = 0.95          # GAE lambda
ENTROPY_COEF          = 0.15 #0.01          # bonus entropia per evitare collasso della policy
VALUE_COEF            = 0.5

# Reward intermedio Stockfish (peso piccolo, si azzera gradualmente)
INTERMEDIATE_REWARD_WEIGHT = 0.1

# Fase freeze: congela backbone + MoveEncoder per i primi N epoch
FREEZE_EPOCHS         = 0 #10

# Learning rates differenziati (usati dopo lo scongelamento)
LR_BACKBONE           = 1e-5
LR_HEADS              = 1e-4


# ---------------------------------------------------------------------------
# Utility: freeze / unfreeze
# ---------------------------------------------------------------------------

def freeze_representation(model: JellyFishPointer):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.move_encoder.parameters():
        p.requires_grad = False
    tqdm.write("  [freeze] backbone + MoveEncoder congelati")


def unfreeze_representation(model: JellyFishPointer, optimizer: Adam):
    for p in model.backbone.parameters():
        p.requires_grad = True
    for p in model.move_encoder.parameters():
        p.requires_grad = True

    # Aggiorna i learning rate nel optimizer
    for group in optimizer.param_groups:
        if group["name"] == "backbone":
            group["lr"] = LR_BACKBONE
        else:
            group["lr"] = LR_HEADS
    tqdm.write("  [unfreeze] backbone + MoveEncoder scongelati con LR differenziato")


def build_optimizer(model: JellyFishPointer) -> Adam:
    """Ottimizzatore con param groups nominati per gestire LR differenziati."""
    return Adam([
        {"params": list(model.backbone.parameters()) +
                   list(model.move_encoder.parameters()),
         "lr": LR_BACKBONE,   # durante freeze il LR non conta, ma deve esistere
         "name": "backbone"},
        {"params": list(model.policy_head.parameters()) +
                   list(model.value_head.parameters()),
         "lr": LR_BACKBONE,
         "name": "heads"},
    ])


# ---------------------------------------------------------------------------
# Stockfish wrapper
# ---------------------------------------------------------------------------

class StockfishPlayer:
    """Wrapper leggero attorno a Stockfish per generare mosse e valutazioni."""

    def __init__(self, path: str, elo: int = 1200):
        self.sf = stockfish.Stockfish(path=path)
        self.sf.set_elo_rating(elo)
        self.sf.set_skill_level(3)

    def get_move(self, board: chess.Board) -> chess.Move | None:
        self.sf.set_fen_position(board.fen())
        uci = self.sf.get_best_move()
        return chess.Move.from_uci(uci) if uci else None

    def evaluate(self, board: chess.Board) -> float:
        """
        Valutazione della posizione in [-1, 1].
        Matto → ±1, centipawn clampati a ±1000.
        """
        self.sf.set_fen_position(board.fen())
        ev = self.sf.get_evaluation()
        if ev["type"] == "mate":
            return 1.0 if ev["value"] > 0 else -1.0
        cp = max(-1000, min(1000, ev["value"]))
        return cp / 1000.0


# ---------------------------------------------------------------------------
# Inferenza: seleziona mossa dalla policy
# ---------------------------------------------------------------------------

@torch.no_grad()
def select_move(
    model: JellyFishPointer,
    board: chess.Board,
    device: torch.device,
    temperature: float = 1.0,
) -> tuple[chess.Move, int, float, float]:
    """
    Seleziona una mossa dalla policy della rete.

    Returns:
        move        : chess.Move selezionata
        move_idx    : indice della mossa nella lista legali
        log_prob    : log probabilità della mossa scelta
        value       : stima del valore della posizione
    """
    legal_moves_list = list(board.legal_moves)
    if not legal_moves_list:
        return None, -1, 0.0, 0.0

    board_tensor  = encode_board(board.fen()).unsqueeze(0).to(device)       # [1, 13, 8, 8]
    moves_tensor  = encode_legal_moves(board).unsqueeze(0).to(device)       # [1, N, 46]

    logits, probs, value = model(board_tensor, moves_tensor)
    # logits: [1, N], value: [1, 1]

    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
        probs  = torch.softmax(logits, dim=-1)

    dist     = torch.distributions.Categorical(probs=probs[0])
    move_idx = dist.sample().item()
    log_prob = dist.log_prob(torch.tensor(move_idx, device=device)).item()

    # Debug distribuzione
    if probs[0].max().item() > 0.9:
        tqdm.write(f"  ATTENZIONE: policy collassata, max prob = {probs[0].max().item():.3f}")

    return legal_moves_list[move_idx], move_idx, log_prob, value[0, 0].item()


# ---------------------------------------------------------------------------
# Generazione partita
# ---------------------------------------------------------------------------

def play_game(
    model: JellyFishPointer,
    sf_player: StockfishPlayer,
    device: torch.device,
    model_plays_white: bool = True,
) -> list[dict]:
    """
    Gioca una partita completa tra il modello e Stockfish.

    Returns:
        trajectory : lista di dict con i campi necessari per PPO,
                     solo per le mosse fatte dal modello
    """
    model.eval()
    board      = chess.Board()
    trajectory = []
    prev_eval  = 0.0

    for move_num in range(MAX_MOVES_PER_GAME):
        if board.is_game_over():
            break

        is_model_turn = (board.turn == chess.WHITE) == model_plays_white

        if is_model_turn:
            move, move_idx, log_prob, value = select_move(model, board, device, temperature=2.0)
            if move is None:
                break

            # Salva il fen PRIMA della mossa — serve per PPO update
            pre_move_fen = board.fen()

            # Valutazione prima della mossa (per reward intermedio)
            eval_before = prev_eval

            board.push(move)

            # Valutazione dopo la mossa
            if not board.is_game_over():
                eval_after = sf_player.evaluate(board)
                # Dal punto di vista del giocatore che ha appena mosso
                if not model_plays_white:
                    eval_after = -eval_after
            else:
                eval_after = 0.0

            intermediate_reward = (eval_after - eval_before) * INTERMEDIATE_REWARD_WEIGHT
            prev_eval = eval_after

            trajectory.append({
                "pre_move_fen": pre_move_fen,  # fen PRIMA della mossa
                "move_idx":     move_idx,
                "log_prob_old": log_prob,
                "value":        value,
                "reward":       intermediate_reward,
                "done":         board.is_game_over(),
            })

        else:
            sf_move = sf_player.get_move(board)
            if sf_move is None or sf_move not in board.legal_moves:
                break
            board.push(sf_move)
            if not board.is_game_over():
                prev_eval = sf_player.evaluate(board)
                if model_plays_white:
                    prev_eval = -prev_eval

    if len(trajectory) > 0:
        tqdm.write(f"  Partita: {len(trajectory)} mosse modello, reward terminale: {trajectory[-1]['reward']:.3f}")
        tqdm.write(f"  Rewards intermedi: min={min(s['reward'] for s in trajectory):.3f} max={max(s['reward'] for s in trajectory):.3f}")

    # Reward terminale
    result = board.result()   # "1-0", "0-1", "1/2-1/2", "*"
    if result == "1-0":
        terminal_reward = 1.0 if model_plays_white else -1.0
    elif result == "0-1":
        terminal_reward = -1.0 if model_plays_white else 1.0
    else:
        terminal_reward = 0.0

    if trajectory:
        trajectory[-1]["reward"] += terminal_reward

    model.train()
    return trajectory


# ---------------------------------------------------------------------------
# GAE — Generalized Advantage Estimation
# ---------------------------------------------------------------------------

def compute_gae(trajectory: list[dict]) -> list[dict]:
    """
    Aggiunge 'advantage' e 'return' a ogni step della traiettoria.
    """
    rewards = [s["reward"] for s in trajectory]
    values  = [s["value"]  for s in trajectory]
    dones   = [s["done"]   for s in trajectory]

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
    """
    Esegue PPO_EPOCHS passate sul buffer raccolto.
    Restituisce le loss medie per il logging.
    """
    # Normalizza i vantaggi sull'intero buffer
    advantages = torch.tensor([s["advantage"] for s in buffer], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_entropy     = 0.0
    n_updates         = 0

    for _ in range(PPO_EPOCHS):
        indices = list(range(len(buffer)))
        random.shuffle(indices)

        for start in range(0, len(buffer), PPO_BATCH_SIZE):
            batch_idx = indices[start:start + PPO_BATCH_SIZE]
            if not batch_idx:
                continue

            # Ricostituisci tensori dal buffer
            board_tensors = []
            moves_tensors = []
            move_indices  = []
            old_log_probs = []
            returns_      = []
            advs          = []

            for idx in batch_idx:
                step = buffer[idx]
                board = chess.Board(step["pre_move_fen"])

                board_tensors.append(encode_board(step["pre_move_fen"]))
                moves_tensors.append(encode_legal_moves(board))
                move_indices.append(step["move_idx"])
                old_log_probs.append(step["log_prob_old"])
                returns_.append(step["return"])
                advs.append(advantages[idx].item())

            # Padding delle mosse (lunghezze variabili nel mini-batch)
            max_n = max(m.shape[0] for m in moves_tensors)
            B     = len(batch_idx)

            moves_padded = torch.zeros(B, max_n, MOVE_VECTOR_DIM, device=device)
            move_mask    = torch.zeros(B, max_n, dtype=torch.bool, device=device)

            for i, m in enumerate(moves_tensors):
                n = m.shape[0]
                moves_padded[i, :n] = m.to(device)
                move_mask[i, :n]    = True

            board_t    = torch.stack(board_tensors).to(device)
            labels_t   = torch.tensor(move_indices,  dtype=torch.long,    device=device)
            old_lp_t   = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
            returns_t  = torch.tensor(returns_,      dtype=torch.float32, device=device).unsqueeze(1)
            adv_t      = torch.tensor(advs,          dtype=torch.float32, device=device)

            # Forward
            logits, probs, value_pred = model(board_t, moves_padded, move_mask)

            # Log probs nuove
            dist         = torch.distributions.Categorical(probs=probs)
            new_log_probs = dist.log_prob(labels_t)
            entropy       = dist.entropy().mean()

            # PPO ratio e clipping
            ratio       = torch.exp(new_log_probs - old_lp_t)
            surr1       = ratio * adv_t
            surr2       = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss  = F.mse_loss(value_pred, returns_t)

            # Loss totale
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_entropy     += entropy.item()
            n_updates         += 1

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


def load_checkpoint(path: str, model: JellyFishPointer, optimizer: Adam, scheduler):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt and scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    epoch        = ckpt.get("epoch", 0)
    best_reward  = ckpt.get("best_avg_reward", -float("inf"))
    tqdm.write(f"  → checkpoint caricato: {path}  (epoch {epoch})")
    return epoch, best_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    model     = JellyFishPointer().to(DEVICE)
    optimizer = build_optimizer(model)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    start_epoch   = 1
    best_reward   = -float("inf")

    # Carica checkpoint RL se esiste, altrimenti supervised
    rl_last = os.path.join(RL_CHECKPOINT_DIR, "last.pt")
    if os.path.exists(rl_last):
        print("Checkpoint RL trovato, riprendo...")
        start_epoch, best_reward = load_checkpoint(rl_last, model, optimizer, scheduler)
        start_epoch += 1
        # Ripristina freeze/unfreeze in base all'epoca
        if start_epoch <= FREEZE_EPOCHS:
            freeze_representation(model)
        else:
            unfreeze_representation(model, optimizer)
    elif os.path.exists(SUPERVISED_CHECKPOINT):
        print(f"Nessun checkpoint RL trovato. Carico supervised: {SUPERVISED_CHECKPOINT}")
        ckpt = torch.load(SUPERVISED_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        freeze_representation(model)
    else:
        print("Nessun checkpoint trovato, parto da zero con freeze attivo.")
        freeze_representation(model)

    sf_player = StockfishPlayer(STOCKFISH_PATH, elo=STOCKFISH_ELO)
    print(f"Stockfish caricato (Elo {STOCKFISH_ELO})\n")
    print(f"Inizio RL training per {EPOCHS} epoche totali (da epoca {start_epoch})\n")

    epoch_bar = tqdm(range(start_epoch, EPOCHS + 1), desc="Epoche RL", dynamic_ncols=True)

    for epoch in epoch_bar:

        # Unfreeze al momento giusto
        if epoch == FREEZE_EPOCHS + 1:
            unfreeze_representation(model, optimizer)

        # ----------------------------------------------------------------
        # Generazione partite
        # ----------------------------------------------------------------
        buffer       = []
        game_rewards = []
        wins = draws = losses = 0

        game_bar = tqdm(range(GAMES_PER_EPOCH), desc=f"  Epoch {epoch:03d} [gioco] ", leave=False, dynamic_ncols=True)

        for game_idx in game_bar:
            model_white = (game_idx % 2 == 0)   # alterna colore
            trajectory  = play_game(model, sf_player, DEVICE, model_plays_white=model_white)

            if not trajectory:
                continue

            trajectory = compute_gae(trajectory)

            # Aggiungi pre_move_fen al buffer (serve per PPO update)
            # Ricostruiamo le posizioni prima di ogni mossa dal fen dopo
            # In realtà salviamo già pre_move_fen dentro play_game — vedi fix sotto
            buffer.extend(trajectory)

            # Statistiche partita
            terminal_r = sum(s["reward"] for s in trajectory)
            game_rewards.append(terminal_r)

            if terminal_r > 0.5:
                wins += 1
            elif terminal_r < -0.5:
                losses += 1
            else:
                draws += 1

            game_bar.set_postfix({
                "W": wins, "D": draws, "L": losses,
                "buf": len(buffer),
            })

        if not buffer:
            tqdm.write(f"Epoch {epoch:03d} | Buffer vuoto, salto aggiornamento.")
            continue

        avg_reward = sum(game_rewards) / len(game_rewards) if game_rewards else 0.0

        # ----------------------------------------------------------------
        # PPO update
        # ----------------------------------------------------------------
        model.train()
        loss_stats = ppo_update(model, optimizer, buffer, DEVICE)
        model.eval()

        scheduler.step(avg_reward)

        tqdm.write(
            f"Epoch {epoch:03d} | "
            f"W/D/L: {wins}/{draws}/{losses}  "
            f"avg_reward: {avg_reward:+.3f}  "
            f"p_loss: {loss_stats['policy_loss']:.4f}  "
            f"v_loss: {loss_stats['value_loss']:.4f}  "
            f"entropy: {loss_stats['entropy']:.4f}  "
            f"LR: {optimizer.param_groups[1]['lr']:.2e}"
        )

        checkpoint_state = {
            "epoch":           epoch,
            "model":           model.state_dict(),
            "optimizer":       optimizer.state_dict(),
            "scheduler":       scheduler.state_dict(),
            "best_avg_reward": best_reward,
        }

        save_checkpoint(checkpoint_state, os.path.join(RL_CHECKPOINT_DIR, "last.pt"))

        if avg_reward > best_reward:
            best_reward = avg_reward
            checkpoint_state["best_avg_reward"] = best_reward
            save_checkpoint(checkpoint_state, os.path.join(RL_CHECKPOINT_DIR, "best.pt"))
            tqdm.write(f"  ★ Nuovo best avg reward: {best_reward:+.3f}")

        epoch_bar.set_postfix({
            "W/D/L":      f"{wins}/{draws}/{losses}",
            "avg_reward": f"{avg_reward:+.3f}",
            "best":       f"{best_reward:+.3f}",
        })

    print(f"\nTraining RL completato. Best avg reward: {best_reward:+.3f}")


if __name__ == "__main__":
    main()
