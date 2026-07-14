"""
train_stockfish.py — Training su dataset Stockfish depth-20.

Dataset atteso (CSV):
    FEN, Evaluation, Move
    - Evaluation: centipawn come '+150', '-80', o mate '#+2', '#-3'
      (dal punto di vista del BIANCO, come da output Stockfish standard)
    - Move: mossa migliore in UCI

Differenze rispetto al training su partite Lichess:
    - Value target: tanh(cp/400) invece di outcome ±1/0
      Molto piu informativo: una posizione +100cp vale ~0.24, non +1.0
    - Policy target: mossa migliore Stockfish (non mossa umana)
    - Eval e' dal punto di vista del bianco -> negata se e' nero a muovere

Fasi:
    1. WARMUP_EPOCHS: MSE(eval) + CE(best move), no Bellman
    2. Da WARMUP_EPOCHS+1: aggiunge Bellman consistency loss

Utilizzo:
    python train_stockfish.py
"""

import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import chess

from MLChess import encode_board, encode_legal_moves, JellyFishPointer, encode_board_fast, encode_board_batch

MOVE_VECTOR_DIM = 46

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

STOCKFISH_CSV  = "stockfish_lichess.csv"   # FEN, Evaluation, Move
CHECKPOINT_IN  = "checkpoints_az_v2/best.pt"    # parte da qui se esiste
CHECKPOINT_DIR = "checkpoints_bellman"
CHECKPOINT_OUT = os.path.join(CHECKPOINT_DIR, "last.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
TOTAL_EPOCHS      = 30
BATCH_SIZE        = 256
LR                = 5e-4
WEIGHT_DECAY      = 1e-4
VALUE_LOSS_WEIGHT = 2.0
GRAD_CLIP         = 1.0

# Conversione centipawn → win probability
CP_SCALE = 400   # tanh(cp / CP_SCALE): a 400cp → 0.96, a 100cp → 0.24

# Bellman
WARMUP_EPOCHS         = 0
BELLMAN_WEIGHT        = 0.1
TAU_START             = 2.0
TAU_FINAL             = 0.5
BELLMAN_SUBSAMPLE     = 64
VALUE_COLLAPSE_THRESH = 0.05

# Dataset
MAX_SAMPLES  = None    # None = usa tutto
VAL_FRACTION = 0.02
NUM_WORKERS  = 4

# ---------------------------------------------------------------------------
# Conversione evaluation
# ---------------------------------------------------------------------------

def eval_to_winprob(eval_str: str, turn: chess.Color) -> float:
    """
    Converte l'evaluation Stockfish (dal punto di vista del bianco)
    in win probability dal punto di vista del giocatore che muove.

    eval_str: '+150', '-80', '#+2', '#-3', '0'
    turn:     chess.WHITE o chess.BLACK
    """
    s = eval_str.strip()

    if s.startswith('#'):
        # Mate score: #+N = bianco da matto in N, #-N = nero da matto in N
        val_white = 1.0 if s.startswith('#+') else -1.0
    else:
        try:
            cp = float(s)
        except ValueError:
            cp = 0.0
        # Clipping conservativo per evitare saturazione a 1.0
        cp = max(-2000.0, min(2000.0, cp))
        val_white = math.tanh(cp / CP_SCALE)

    # Flip se e' nero a muovere: il value head predice sempre
    # dal punto di vista del giocatore che muove
    return val_white if turn == chess.WHITE else -val_white


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StockfishDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        fen      = str(row["FEN"])
        eval_str = str(row["Evaluation"])
        move_uci = str(row["Move"])

        board      = chess.Board(fen)
        legal_list = list(board.legal_moves)

        board_t = encode_board_fast(fen)
        moves_t = encode_legal_moves(board)

        # Value target: centipawn → win probability (player-to-move perspective)
        value = eval_to_winprob(eval_str, board.turn)

        # Policy target: one-hot sulla mossa migliore Stockfish
        target_vec = torch.zeros(len(legal_list))
        try:
            best = chess.Move.from_uci(move_uci)
            if best in legal_list:
                target_vec[legal_list.index(best)] = 1.0
            else:
                target_vec[0] = 1.0
        except Exception:
            target_vec[0] = 1.0

        return {
            "board_t":  board_t,
            "moves_t":  moves_t,
            "policy_t": target_vec,
            "value_t":  torch.tensor([value], dtype=torch.float32),
            "n_moves":  len(legal_list),
            "fen":      fen,
        }


def collate_fn(batch):
    max_n = max(item["n_moves"] for item in batch)
    B     = len(batch)

    boards_t      = torch.stack([item["board_t"] for item in batch])
    moves_padded  = torch.zeros(B, max_n, MOVE_VECTOR_DIM)
    move_mask     = torch.zeros(B, max_n, dtype=torch.bool)
    policy_padded = torch.zeros(B, max_n)
    values_t      = torch.stack([item["value_t"] for item in batch])
    fens          = [item["fen"] for item in batch]

    for i, item in enumerate(batch):
        n = item["n_moves"]
        moves_padded[i, :n]  = item["moves_t"]
        move_mask[i, :n]     = True
        policy_padded[i, :n] = item["policy_t"]

    return {
        "boards_t":     boards_t,
        "moves_padded": moves_padded,
        "move_mask":    move_mask,
        "policy_padded":policy_padded,
        "values_t":     values_t,
        "fens":         fens,
    }


# ---------------------------------------------------------------------------
# Generazione board figli (lazy)
# ---------------------------------------------------------------------------

def build_children_batch(fens: list, device: torch.device):
    all_children = []
    all_n = []
    max_n = 0

    for fen in fens:
        board = chess.Board(fen)

        child_fens = []

        for move in board.legal_moves:
            board.push(move)
            child_fens.append(board.fen())
            board.pop()

        if len(child_fens) > 0:
            children = encode_board_batch(child_fens)
        else:
            children = torch.empty((0, 13, 8, 8), dtype=torch.float32)

        all_children.append(children)
        all_n.append(len(child_fens))
        max_n = max(max_n, len(child_fens))

    B = len(fens)

    children_padded = torch.zeros(B, max_n, 13, 8, 8)
    child_mask = torch.zeros(B, max_n, dtype=torch.bool)

    for i, (children, n) in enumerate(zip(all_children, all_n)):
        if n > 0:
            children_padded[i, :n] = children
            child_mask[i, :n] = True

    return children_padded.to(device), child_mask.to(device)


# ---------------------------------------------------------------------------
# Bellman consistency loss
# ---------------------------------------------------------------------------

def bellman_consistency_loss(model, children_padded, child_mask, value_parent, tau):
    """
    L = mean( V(s) + softmin_tau V(s') )^2

    Usa log-mean-exp (non log-sum-exp) per evitare il bias tau*log(n).
    """
    B_sub, max_n, C, H, W = children_padded.shape

    children_flat = children_padded.view(B_sub * max_n, C, H, W)
    h_children    = model.backbone(children_flat)
    v_children    = model.value_head(h_children).view(B_sub, max_n)

    n_real   = child_mask.sum(dim=1).float().clamp(min=1.0)
    INF      = torch.finfo(v_children.dtype).max / 2
    v_masked = v_children.masked_fill(~child_mask, INF)

    soft_min = -tau * torch.logsumexp(-v_masked / tau, dim=1) \
               + tau * torch.log(n_real)

    residual = value_parent.squeeze(1) + soft_min
    return (residual ** 2).mean()


# ---------------------------------------------------------------------------
# Utilita
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_value_std(v: torch.Tensor) -> float:
    return v.squeeze().float().std().item()


def bellman_tau(epoch: int) -> float:
    if epoch <= WARMUP_EPOCHS:
        return TAU_START
    progress = (epoch - WARMUP_EPOCHS) / max(TOTAL_EPOCHS - WARMUP_EPOCHS, 1)
    return TAU_START + progress * (TAU_FINAL - TAU_START)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch":     epoch,
        "val_loss":  val_loss,
    }, tmp)
    os.replace(tmp, path)
    tqdm.write(f"  -> checkpoint: {path}  (epoch {epoch}, val_loss {val_loss:.4f})")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=DEVICE)
    sd   = ckpt.get("model", ckpt)
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    if optimizer and "optimizer" in ckpt:
        try:    optimizer.load_state_dict(ckpt["optimizer"])
        except Exception: pass
    if scheduler and "scheduler" in ckpt and ckpt["scheduler"]:
        try:    scheduler.load_state_dict(ckpt["scheduler"])
        except Exception: pass
    epoch    = ckpt.get("epoch", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    tqdm.write(f"  -> caricato: {path}  (epoch {epoch})")
    return epoch, val_loss


# ---------------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device,
              train=True, use_bellman=False, tau=TAU_START):

    model.train() if train else model.eval()
    for p in model.parameters(): p.requires_grad = train

    total_p, total_v, total_b, total_acc = 0.0, 0.0, 0.0, 0.0
    vstd_list, n_batches, bellman_skipped = [], 0, 0

    ctx = torch.no_grad() if not train else torch.enable_grad()

    with ctx:
        for batch in tqdm(loader, leave=False):
            boards_t      = batch["boards_t"].to(device)
            moves_padded  = batch["moves_padded"].to(device)
            move_mask     = batch["move_mask"].to(device)
            policy_padded = batch["policy_padded"].to(device)
            values_t      = batch["values_t"].to(device)
            fens          = batch["fens"]

            _, probs, value_pred = model(boards_t, moves_padded, move_mask)

            log_probs   = torch.log(probs + 1e-8)
            policy_loss = -(policy_padded * log_probs).sum(dim=1).mean()
            value_loss  = F.mse_loss(value_pred, values_t)
            loss        = policy_loss + VALUE_LOSS_WEIGHT * value_loss

            b_loss_val = torch.tensor(0.0, device=device)
            if train and use_bellman:
                vstd = compute_value_std(value_pred)
                if vstd < VALUE_COLLAPSE_THRESH:
                    bellman_skipped += 1
                else:
                    has_children = move_mask.any(dim=1)
                    valid_idx    = has_children.nonzero(as_tuple=True)[0]
                    if len(valid_idx) > 0:
                        sub_idx  = valid_idx[
                            torch.randperm(len(valid_idx), device=device)[:BELLMAN_SUBSAMPLE]
                        ]
                        sub_fens = [fens[i] for i in sub_idx.cpu().tolist()]
                        children_padded, child_mask = build_children_batch(sub_fens, device)
                        b_loss_val = bellman_consistency_loss(
                            model, children_padded, child_mask,
                            value_pred[sub_idx], tau,
                        )
                        loss = loss + BELLMAN_WEIGHT * b_loss_val

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                optimizer.step()

            with torch.no_grad():
                vstd_list.append(compute_value_std(value_pred))
                pred_idx   = probs.argmax(dim=1)
                target_idx = policy_padded.argmax(dim=1)
                total_acc += (pred_idx == target_idx).float().mean().item()

            total_p += policy_loss.item()
            total_v += value_loss.item()
            total_b += b_loss_val.item()
            n_batches += 1

    d = max(n_batches, 1)
    return {
        "policy_loss":     total_p   / d,
        "value_loss":      total_v   / d,
        "bellman_loss":    total_b   / d,
        "accuracy":        total_acc / d,
        "value_std":       sum(vstd_list) / max(len(vstd_list), 1),
        "bellman_skipped": bellman_skipped,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print(f"Dataset: {STOCKFISH_CSV}")
    print(f"Warmup (no Bellman): epoche 1-{WARMUP_EPOCHS}")
    print(f"Bellman (lambda={BELLMAN_WEIGHT}): epoche {WARMUP_EPOCHS+1}-{TOTAL_EPOCHS}")
    print(f"tau: {TAU_START} -> {TAU_FINAL}\n")

    df = pd.read_csv(STOCKFISH_CSV)
    print(f"Posizioni raw: {len(df):,}")

    # Pulizia: rimuovi righe con valori mancanti
    df = df.dropna(subset=["FEN", "Evaluation", "Move"])
    df = df[df["Evaluation"].astype(str).str.strip() != ""]
    print(f"Posizioni dopo pulizia: {len(df):,}")

    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42)
        print(f"Campionamento a {MAX_SAMPLES:,}")

    # Distribuzione eval (quante mate, quante cp)
    evals = df["Evaluation"].astype(str)
    n_mate = evals.str.startswith('#').sum()
    print(f"Mate positions: {n_mate:,} ({100*n_mate/len(df):.1f}%)")
    print(f"CP positions:   {len(df)-n_mate:,}\n")

    val_size = max(1000, int(len(df) * VAL_FRACTION))
    val_df   = df.sample(n=val_size, random_state=42)
    train_df = df.drop(val_df.index)
    print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}\n")

    train_loader = DataLoader(
        StockfishDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        StockfishDataset(val_df), batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = JellyFishPointer().to(DEVICE)

    if os.path.exists(CHECKPOINT_IN):
        ckpt = torch.load(CHECKPOINT_IN, map_location=DEVICE)
        sd   = ckpt.get("model", ckpt)
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"Checkpoint caricato: {CHECKPOINT_IN}")
    else:
        print("Nessun checkpoint — parto da zero.")

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler     = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)
    best_val_loss = float("inf")
    start_epoch   = 1

    if os.path.exists(CHECKPOINT_OUT):
        start_epoch, best_val_loss = load_checkpoint(
            CHECKPOINT_OUT, model, optimizer, scheduler
        )
        start_epoch += 1

    print(f"Parametri: {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        use_bellman = epoch > WARMUP_EPOCHS
        tau         = bellman_tau(epoch)
        phase       = "warmup          " if not use_bellman else f"bellman tau={tau:.2f}"

        train_s = run_epoch(model, train_loader, optimizer, DEVICE,
                            train=True, use_bellman=use_bellman, tau=tau)
        val_s   = run_epoch(model, val_loader, optimizer, DEVICE,
                            train=False, use_bellman=False)

        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{TOTAL_EPOCHS} [{phase}] | "
            f"train  p:{train_s['policy_loss']:.4f} v:{train_s['value_loss']:.4f} "
            f"b:{train_s['bellman_loss']:.4f} acc:{train_s['accuracy']:.3f} "
            f"vstd:{train_s['value_std']:.3f}  |  "
            f"val  p:{val_s['policy_loss']:.4f} v:{val_s['value_loss']:.4f} "
            f"acc:{val_s['accuracy']:.3f} vstd:{val_s['value_std']:.3f}"
        )

        if train_s["bellman_skipped"] > 0:
            print(f"  ATTENZIONE collasso: Bellman saltato in "
                  f"{train_s['bellman_skipped']} batch (vstd < {VALUE_COLLAPSE_THRESH})")

        gap = val_s["policy_loss"] - train_s["policy_loss"]
        if gap > 0.8:
            print(f"  ATTENZIONE overfitting: gap train/val = {gap:.3f}")

        save_checkpoint(model, optimizer, scheduler, epoch,
                        val_s["policy_loss"], CHECKPOINT_OUT)
        if val_s["policy_loss"] < best_val_loss:
            best_val_loss = val_s["policy_loss"]
            save_checkpoint(model, optimizer, scheduler, epoch,
                            best_val_loss, os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  * Nuovo best: {best_val_loss:.4f}")

    print(f"\nCompletato. Best val policy_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()