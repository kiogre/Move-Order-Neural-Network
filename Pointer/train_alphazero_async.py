"""
train_alphazero_async.py — AlphaZero asincrono con Inference Server, Actor, Learner.

Architettura:
  - 1 Inference Server (GPU): raccoglie foglie dagli Actor, forward pass batched, restituisce risultati
  - 1 Learner (GPU): training sul replay buffer condiviso, aggiorna pesi ogni N step
  - N Actor (CPU): self-play con MCTS, mandano foglie all'Inference Server

REGOLA FONDAMENTALE: nelle Queue non passano mai tensori PyTorch.
Tutto viaggia come numpy array o tipi Python puri (str, int, float, list).
I tensori vengono creati solo dentro il processo che li usa.
"""

import os
import math
import time
import random
import resource
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from collections import deque
from typing import Optional

from MLChess import encode_board, encode_legal_moves, JellyFishPointer

MOVE_VECTOR_DIM = 46

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

CHECKPOINT_PATH      = "checkpoints_az_v2/last.pt"
ASYNC_CHECKPOINT_DIR = "checkpoints_az_async"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_ACTORS             = 4
GAMES_PER_ACTOR      = 16
NUM_SIMULATIONS      = 50
MAX_MOVES            = 150
TEMP_HIGH            = 1.0
TEMP_LOW             = 0.01
TEMP_THRESHOLD       = 10
CURRICULUM_PROB      = 0.30
CURRICULUM_MAX_MOVES = 120

INFERENCE_BATCH_SIZE   = 256
INFERENCE_TIMEOUT      = 0.005

BUFFER_SIZE            = 200_000
MIN_BUFFER             = 1_000
BATCH_SIZE             = 256
LR_BACKBONE            = 1e-5
LR_HEADS               = 1e-4
VALUE_LOSS_WEIGHT      = 3.0
MIXED_BUFFER_RATIO     = 0.20
WEIGHT_UPDATE_INTERVAL = 50
CURRICULUM_CSV         = "../over_mate_1_tactic_evals.csv"

# ---------------------------------------------------------------------------
# Nodo MCTS
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = (
        "parent", "move", "_board", "prior",
        "children", "visit_count", "value_sum",
        "is_expanded", "is_terminal",
        "_board_np", "_legal_moves",
    )

    def __init__(self, board, prior=1.0, parent=None, move=None):
        self._board       = board
        self.parent       = parent
        self.move         = move
        self.prior        = prior
        self.children     = {}
        self.visit_count  = 0
        self.value_sum    = 0.0
        self.is_expanded  = False
        self.is_terminal  = board.is_game_over()
        self._board_np    = None
        self._legal_moves = None

    @property
    def board(self): return self._board

    @property
    def legal_moves(self):
        if self._legal_moves is None:
            self._legal_moves = list(self._board.legal_moves)
        return self._legal_moves

    def get_board_np(self) -> np.ndarray:
        if self._board_np is None:
            self._board_np = encode_board(self._board.fen()).numpy()
        return self._board_np

    @property
    def Q(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def puct_score(self, c_puct, fpu_value=-0.1):
        assert self.parent is not None
        q = fpu_value if self.visit_count == 0 else self.Q
        u = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return q + u

    def best_child(self, c_puct, fpu_value=-0.1):
        return max(self.children.values(), key=lambda n: n.puct_score(c_puct, fpu_value))


# ---------------------------------------------------------------------------
# Stato partita
# ---------------------------------------------------------------------------

class GameState:
    def __init__(self, game_idx, main_is_white, start_fen=None):
        self.game_idx      = game_idx
        self.main_is_white = main_is_white
        self.board         = chess.Board(start_fen) if start_fen else chess.Board()
        self.steps         = []
        self.move_num      = 0
        self.done          = False
        self.result        = None
        self.root: Optional[MCTSNode] = None
        self.sims_done     = 0

    @property
    def max_moves(self):
        return CURRICULUM_MAX_MOVES if self.board.fen().split()[0] != chess.Board().fen().split()[0] else MAX_MOVES


# ---------------------------------------------------------------------------
# INFERENCE SERVER
# ---------------------------------------------------------------------------

def inference_server(
    model_state_dict,
    leaf_queue:    mp.Queue,
    result_queues: list,
    weight_queue:  mp.Queue,
    stop_event:    mp.Event,
):
    print("[InferenceServer] Avvio...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = JellyFishPointer().to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    pending = []  # (actor_id, req_id, board_np, moves_np, n_moves)

    while not stop_event.is_set():

        # Aggiorna pesi
        while not weight_queue.empty():
            try:
                new_sd = weight_queue.get_nowait()
                model.load_state_dict(new_sd)
                model.eval()
            except Exception:
                pass

        # Raccogli foglie
        deadline = time.monotonic() + INFERENCE_TIMEOUT
        while time.monotonic() < deadline and len(pending) < INFERENCE_BATCH_SIZE:
            try:
                item = leaf_queue.get(timeout=INFERENCE_TIMEOUT / 4)
                pending.append(item)
            except Exception:
                break

        if not pending:
            continue

        actor_ids = [r[0] for r in pending]
        req_ids   = [r[1] for r in pending]
        board_nps = [r[2] for r in pending]
        moves_nps = [r[3] for r in pending]
        n_moves   = [r[4] for r in pending]

        max_n = max(n_moves)
        B     = len(pending)

        boards_t     = torch.from_numpy(np.stack(board_nps)).to(device)
        moves_padded = torch.zeros(B, max_n, MOVE_VECTOR_DIM, device=device)
        move_mask    = torch.zeros(B, max_n, dtype=torch.bool, device=device)

        for i, (m_np, n) in enumerate(zip(moves_nps, n_moves)):
            moves_padded[i, :n] = torch.from_numpy(m_np[:n]).to(device)
            move_mask[i, :n]    = True

        with torch.no_grad():
            _, probs, value_pred = model(boards_t, moves_padded, move_mask)

        # Restituisci come numpy — mai tensori nelle Queue
        for i in range(B):
            probs_np = probs[i, :n_moves[i]].cpu().numpy()
            value    = float(value_pred[i, 0].item())
            result_queues[actor_ids[i]].put((req_ids[i], probs_np, value))

        pending.clear()

    print("[InferenceServer] Stop.")


# ---------------------------------------------------------------------------
# ACTOR
# ---------------------------------------------------------------------------

def actor_process(
    actor_id:        int,
    leaf_queue:      mp.Queue,
    result_queue:    mp.Queue,
    buffer_queue:    mp.Queue,
    stop_event:      mp.Event,
    curriculum_fens: list,
    c_puct:          float = 2.5,
    fpu_value:       float = -0.1,
):
    print(f"[Actor {actor_id}] Avvio ({GAMES_PER_ACTOR} partite)...")
    req_counter = 0

    def next_req():
        nonlocal req_counter
        req_counter += 1
        return req_counter

    def select_leaf(root):
        node = root
        while node.is_expanded and not node.is_terminal:
            if not node.children:
                break
            node = node.best_child(c_puct, fpu_value)
        return node

    def backpropagate(node, value):
        cur = node
        while cur is not None:
            cur.visit_count += 1
            cur.value_sum   += value
            value = -value
            cur   = cur.parent

    def expand_node(node, probs_np):
        legal = node.legal_moves
        if not legal:
            node.is_terminal = True
            return
        node.is_expanded = True
        for j, move in enumerate(legal):
            cb = node.board.copy()
            cb.push(move)
            node.children[move] = MCTSNode(
                board  = cb,
                prior  = float(probs_np[j]) if j < len(probs_np) else 1e-8,
                parent = node,
                move   = move,
            )

    def add_dirichlet(root):
        if not root.children:
            return
        children = list(root.children.values())
        noise    = np.random.dirichlet([0.3] * len(children))
        for child, n in zip(children, noise):
            child.prior = 0.75 * child.prior + 0.25 * float(n)

    def terminal_value(node):
        outcome = node.board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner != node.board.turn else -1.0

    def send_leaf(node):
        req_id   = next_req()
        board_np = node.get_board_np()
        moves_np = encode_legal_moves(node.board).numpy()
        leaf_queue.put((actor_id, req_id, board_np, moves_np, len(node.legal_moves)))
        return req_id

    def recv_result(expected_req_id, stash):
        if expected_req_id in stash:
            return stash.pop(expected_req_id)
        while True:
            req_id, probs_np, value = result_queue.get()
            if req_id == expected_req_id:
                return probs_np, value
            stash[req_id] = (probs_np, value)

    def expand_root(node, stash):
        req_id       = send_leaf(node)
        probs_np, _  = recv_result(req_id, stash)
        expand_node(node, probs_np)
        node.visit_count = 1
        add_dirichlet(node)

    def frozen_move_greedy(board, stash):
        legal = list(board.legal_moves)
        if not legal:
            return None
        req_id   = next_req()
        board_np = encode_board(board.fen()).numpy()
        moves_np = encode_legal_moves(board).numpy()
        leaf_queue.put((actor_id, req_id, board_np, moves_np, len(legal)))
        probs_np, _ = recv_result(req_id, stash)
        return legal[int(np.argmax(probs_np))]

    def finalize(g):
        if not g.done:
            g.done   = True
            g.result = g.board.result()

    def game_terminal(g):
        r = g.result or g.board.result()
        if r == "1-0":   return  1.0 if g.main_is_white else -1.0
        elif r == "0-1": return -1.0 if g.main_is_white else  1.0
        return 0.0

    epoch = 0
    while not stop_event.is_set():
        epoch += 1

        games = []
        for i in range(GAMES_PER_ACTOR):
            fen = random.choice(curriculum_fens) if curriculum_fens and random.random() < CURRICULUM_PROB else None
            games.append(GameState(i, main_is_white=(i % 2 == 0), start_fen=fen))

        stash: dict = {}

        # Espandi radici
        for g in games:
            if g.board.is_game_over() or not list(g.board.legal_moves):
                finalize(g)
                continue
            g.root = MCTSNode(g.board.copy())
            expand_root(g.root, stash)

        # Loop mosse
        while True:
            active = [g for g in games if not g.done]
            if not active:
                break

            # Mosse frozen
            for g in active:
                if g.done:
                    continue
                is_main = (g.board.turn == chess.WHITE) == g.main_is_white
                if is_main:
                    continue
                if g.board.is_game_over() or g.move_num >= g.max_moves:
                    finalize(g)
                    continue
                move = frozen_move_greedy(g.board, stash)
                if move is None:
                    finalize(g)
                    continue
                if g.root is not None and move in g.root.children:
                    g.root = g.root.children[move]
                    g.root.parent = None
                else:
                    g.root = None
                g.board.push(move)
                g.move_num += 1
                if g.board.is_game_over() or g.move_num >= g.max_moves:
                    finalize(g)
                elif g.root is None:
                    g.root = MCTSNode(g.board.copy())
                    expand_root(g.root, stash)

            # Simulazioni MCTS
            need_sims = [
                g for g in games
                if not g.done
                and (g.board.turn == chess.WHITE) == g.main_is_white
                and g.sims_done < NUM_SIMULATIONS
                and not g.board.is_game_over()
            ]

            if need_sims:
                sim_map: dict = {}
                for g in need_sims:
                    leaf = select_leaf(g.root)
                    if leaf.is_terminal:
                        backpropagate(leaf, terminal_value(leaf))
                        g.sims_done += 1
                        continue
                    if leaf.is_expanded:
                        backpropagate(leaf, leaf.Q)
                        g.sims_done += 1
                        continue
                    req_id = send_leaf(leaf)
                    sim_map[req_id] = (g.game_idx, leaf)

                received = 0
                while received < len(sim_map):
                    req_id, probs_np, value = result_queue.get()
                    if req_id in sim_map:
                        gidx, leaf = sim_map[req_id]
                        expand_node(leaf, probs_np)
                        backpropagate(leaf, value)
                        games[gidx].sims_done += 1
                        received += 1
                    else:
                        stash[req_id] = (probs_np, value)
                continue

            # Scegli mosse
            ready = [
                g for g in games
                if not g.done
                and (g.board.turn == chess.WHITE) == g.main_is_white
            ]
            for g in ready:
                if g.board.is_game_over() or g.move_num >= g.max_moves:
                    finalize(g)
                    continue
                if not g.root or not g.root.children:
                    finalize(g)
                    continue

                temp   = TEMP_HIGH if g.move_num < TEMP_THRESHOLD else TEMP_LOW
                visits = np.array([c.visit_count for c in g.root.children.values()], dtype=np.float64)
                moves  = list(g.root.children.keys())

                if temp <= 0.0:
                    move = moves[int(np.argmax(visits))]
                else:
                    visits    = visits ** (1.0 / temp)
                    probs_arr = visits / visits.sum()
                    move      = moves[np.random.choice(len(moves), p=probs_arr)]

                legal_list = g.root.legal_moves
                legal_np   = encode_legal_moves(g.board).numpy()
                t          = max(temp, 1e-8)
                visits_d   = {m: c.visit_count for m, c in g.root.children.items()}
                total_v    = sum(v ** (1.0 / t) for v in visits_d.values()) or 1e-8
                policy_np  = np.array(
                    [(visits_d.get(m, 0) ** (1.0 / t)) / total_v for m in legal_list],
                    dtype=np.float32,
                )
                s = policy_np.sum()
                if s > 0:
                    policy_np /= s

                g.steps.append({
                    "board_fen":      g.board.fen(),
                    "legal_moves_np": legal_np,
                    "policy_np":      policy_np,
                    "value_target":   None,
                })

                if move in g.root.children:
                    g.root = g.root.children[move]
                    g.root.parent = None
                else:
                    g.root = None

                g.board.push(move)
                g.move_num  += 1
                g.sims_done  = 0

                if g.board.is_game_over() or g.move_num >= g.max_moves:
                    finalize(g)
                elif g.root is None:
                    g.root = MCTSNode(g.board.copy())
                    expand_root(g.root, stash)

        # Fine epoch
        all_steps = []
        for g in games:
            terminal = game_terminal(g)
            for step in g.steps:
                step["value_target"] = terminal
            all_steps.extend(g.steps)

        if all_steps:
            buffer_queue.put(all_steps)

        print(f"[Actor {actor_id}] Epoch {epoch}: {len(all_steps)} step")


# ---------------------------------------------------------------------------
# LEARNER
# ---------------------------------------------------------------------------

def learner_process(
    initial_state_dict,
    buffer_queue:       mp.Queue,
    weight_queue:       mp.Queue,
    stop_event:         mp.Event,
    curriculum_samples: list,
):
    print("[Learner] Avvio...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = JellyFishPointer().to(device)
    model.load_state_dict(initial_state_dict)

    optimizer = torch.optim.Adam([
        {"params": list(model.backbone.parameters()) +
                   list(model.move_encoder.parameters()),  "lr": LR_BACKBONE},
        {"params": list(model.policy_head.parameters()) +
                   list(model.value_head.parameters()),    "lr": LR_HEADS},
    ])

    replay_buffer: deque = deque(maxlen=BUFFER_SIZE)
    train_step = 0
    sync_epoch = 0

    while not stop_event.is_set():

        # Raccogli step
        drained = 0
        while not buffer_queue.empty() and drained < 10:
            try:
                steps = buffer_queue.get_nowait()
                replay_buffer.extend(steps)
                drained += 1
            except Exception:
                break

        if len(replay_buffer) < MIN_BUFFER:
            time.sleep(1.0)
            continue

        model.train()

        n_curriculum = int(BATCH_SIZE * MIXED_BUFFER_RATIO) if curriculum_samples else 0
        n_buffer     = BATCH_SIZE - n_curriculum

        batch = random.sample(list(replay_buffer), min(n_buffer, len(replay_buffer)))
        if n_curriculum > 0 and curriculum_samples:
            batch = batch + random.sample(curriculum_samples, min(n_curriculum, len(curriculum_samples)))

        board_tensors  = []
        moves_tensors  = []
        policy_targets = []
        value_targets  = []

        for step in batch:
            board_tensors.append(encode_board(step["board_fen"]))
            lm = step["legal_moves_np"]
            moves_tensors.append(torch.from_numpy(lm) if isinstance(lm, np.ndarray) else lm)
            pt = step["policy_np"]
            policy_targets.append(torch.from_numpy(pt) if isinstance(pt, np.ndarray) else pt)
            value_targets.append(step["value_target"])

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

        boards_t     = torch.stack(board_tensors).to(device)
        values_clean = [v if v is not None else 0.0 for v in value_targets]
        values_t     = torch.tensor(values_clean, dtype=torch.float32, device=device).unsqueeze(1)

        value_mask_t = torch.ones(B, dtype=torch.bool, device=device)
        value_mask_t[n_buffer:] = False

        logits, probs, value_pred = model(boards_t, moves_padded, move_mask)

        log_probs   = torch.log(probs + 1e-8)
        policy_loss = -(policy_padded * log_probs).sum(dim=1).mean()

        if value_mask_t.any():
            value_loss = F.mse_loss(value_pred[value_mask_t], values_t[value_mask_t])
        else:
            value_loss = torch.tensor(0.0, device=device)

        loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_step += 1
        model.eval()

        if train_step % WEIGHT_UPDATE_INTERVAL == 0:
            sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            try:
                weight_queue.put_nowait(sd)
            except Exception:
                pass  # queue piena, salta

            sync_epoch += 1
            print(f"[Learner] Step {train_step} | sync {sync_epoch} | "
                  f"p_loss: {policy_loss.item():.4f} | v_loss: {value_loss.item():.4f} | "
                  f"buf: {len(replay_buffer)}")

            if sync_epoch % 10 == 0:
                os.makedirs(ASYNC_CHECKPOINT_DIR, exist_ok=True)
                ckpt_path = os.path.join(ASYNC_CHECKPOINT_DIR, "last.pt")
                tmp_path  = ckpt_path + ".tmp"
                torch.save({
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "train_step": train_step,
                    "sync_epoch": sync_epoch,
                }, tmp_path)
                os.replace(tmp_path, ckpt_path)
                print(f"[Learner] Checkpoint salvato (step {train_step})")

    print("[Learner] Stop.")


# ---------------------------------------------------------------------------
# Curriculum loading — tutto numpy
# ---------------------------------------------------------------------------

def load_curriculum_fens(csv_path, max_samples=50_000):
    try:
        import pandas as pd
        df = pd.read_csv(csv_path).dropna(subset=["FEN"])
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
        return df["FEN"].tolist()
    except Exception as e:
        print(f"WARNING: curriculum FEN non caricati ({e})")
        return []


def load_curriculum_samples(csv_path, max_samples=50_000):
    try:
        import pandas as pd
        df = pd.read_csv(csv_path).dropna(subset=["FEN", "Move"])
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
        samples = []
        for _, row in df.iterrows():
            try:
                board      = chess.Board(row["FEN"])
                legal_list = list(board.legal_moves)
                if not legal_list:
                    continue
                target_move = chess.Move.from_uci(str(row["Move"]))
                legal_np    = encode_legal_moves(board).numpy()
                target_np   = np.zeros(len(legal_list), dtype=np.float32)
                if target_move in legal_list:
                    target_np[legal_list.index(target_move)] = 1.0
                else:
                    target_np[0] = 1.0
                samples.append({
                    "board_fen":      row["FEN"],
                    "legal_moves_np": legal_np,
                    "policy_np":      target_np,
                    "value_target":   None,
                })
            except Exception:
                continue
        print(f"  Curriculum samples caricati: {len(samples)}")
        return samples
    except Exception as e:
        print(f"WARNING: curriculum samples non caricati ({e})")
        return []


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # Alza limite file descriptor prima di tutto il resto
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft   = min(hard, 65536)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(f"File descriptor limit: {soft} → {new_soft}")
    except Exception as e:
        print(f"WARNING: impossibile alzare fd limit ({e})")

    mp.set_start_method("spawn", force=True)

    print(f"Device: {DEVICE}")
    print(f"Avvio con {N_ACTORS} Actor, {GAMES_PER_ACTOR} partite per Actor")

    model = JellyFishPointer()
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Carico checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        sd   = ckpt["model"]
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
    else:
        print("Nessun checkpoint trovato, parto da zero.")

    # Clone esplicito — nessuna shared memory
    initial_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    del model

    print("Caricamento curriculum...")
    curriculum_fens    = load_curriculum_fens(CURRICULUM_CSV)
    curriculum_samples = load_curriculum_samples(CURRICULUM_CSV)
    print(f"  {len(curriculum_fens)} FEN per Actor, {len(curriculum_samples)} campioni per Learner")

    leaf_queue    = mp.Queue(maxsize=2048)
    result_queues = [mp.Queue(maxsize=512) for _ in range(N_ACTORS)]
    buffer_queue  = mp.Queue(maxsize=256)
    weight_queue  = mp.Queue(maxsize=2)

    stop_event = mp.Event()
    processes  = []

    for name, target, args in [
        ("InferenceServer", inference_server,
         (initial_sd, leaf_queue, result_queues, weight_queue, stop_event)),
        ("Learner", learner_process,
         (initial_sd, buffer_queue, weight_queue, stop_event, curriculum_samples)),
    ]:
        p = mp.Process(target=target, args=args, name=name, daemon=True)
        p.start()
        processes.append(p)
        print(f"[Main] {name} avviato (pid {p.pid})")

    for actor_id in range(N_ACTORS):
        p = mp.Process(
            target = actor_process,
            args   = (actor_id, leaf_queue, result_queues[actor_id],
                      buffer_queue, stop_event, curriculum_fens),
            name   = f"Actor_{actor_id}",
            daemon = True,
        )
        p.start()
        processes.append(p)
        print(f"[Main] Actor {actor_id} avviato (pid {p.pid})")

    print("\nTutti i processi avviati. Ctrl+C per fermare.\n")

    try:
        while True:
            time.sleep(10)
            dead = [p.name for p in processes if not p.is_alive()]
            if dead:
                print(f"WARNING: processi morti: {dead}")
            if not any(p.is_alive() for p in processes):
                break
    except KeyboardInterrupt:
        print("\nInterruzione richiesta...")
        stop_event.set()
        for p in processes:
            p.join(timeout=15)
        print("Stop completato.")


if __name__ == "__main__":
    main()
