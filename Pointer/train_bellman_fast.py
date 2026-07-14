"""
train_bellman_fast.py — Training veloce con child FEN pre-calcolati.

Fix rispetto alla versione precedente:
  - ds_idx ora usa l'indice originale della CSV (non quello post-split)
    cosi il lookup nel memmap e' sempre corretto
  - FEN filtrate correttamente in get_children (n_valid invece di n)
  - Import duplicato rimosso
  - get_children robusto a FEN vuote/malformate

Prerequisito:
  python precompute_child_fens.py --csv stockfish_lichess.csv --output child_fens/

Utilizzo:
  python train_bellman_fast.py
"""

import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import chess

from MLChess import encode_legal_moves, JellyFishPointer
from encode_fast import encode_board_fast, encode_board_batch

MOVE_VECTOR_DIM = 46

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

STOCKFISH_CSV  = "stockfish_lichess.csv"
CHILD_FENS_DIR = "child_fens"
CHECKPOINT_IN  = "checkpoints_bellman/best.pt"
CHECKPOINT_DIR = "checkpoints_bellman_fast"
CHECKPOINT_OUT = os.path.join(CHECKPOINT_DIR, "last.pt")

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True

# Training
TOTAL_EPOCHS      = 30
BATCH_SIZE        = 256
LR                = 5e-4
WEIGHT_DECAY      = 1e-4
VALUE_LOSS_WEIGHT = 2.0
GRAD_CLIP         = 1.0
CP_SCALE          = 400

# Bellman
WARMUP_EPOCHS         = 0
BELLMAN_WEIGHT        = 0.1
TAU_START             = 2.0
TAU_FINAL             = 0.5
BELLMAN_SUBSAMPLE     = 64
VALUE_COLLAPSE_THRESH = 0.05

# Dataset
MAX_SAMPLES  = None
VAL_FRACTION = 0.02
NUM_WORKERS  = 6

# ---------------------------------------------------------------------------
# ChildFenStore
# ---------------------------------------------------------------------------

class ChildFenStore:
    def __init__(self, store_dir: str):
        meta_path = os.path.join(store_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Child FEN store non trovato: {store_dir}\n"
                f"Esegui: python precompute_child_fens.py"
            )
        with open(meta_path) as f:
            meta = json.load(f)

        self.max_children = meta["max_children"]
        self.fen_max_len  = meta["fen_max_len"]

        print(f"Caricamento child FEN store: {store_dir}")
        print(f"  Posizioni: {meta['n_positions']:,}  max_children: {self.max_children}")

        self.child_fens_mm = np.load(meta["child_fens_path"], mmap_mode="r")
        self.n_children_mm = np.load(meta["n_children_path"], mmap_mode="r")

    def get_children(self, indices: list, device: torch.device):
        """
        Carica child FEN per gli indici dati (indici nella CSV originale).
        Usa n_valid = len(fens_valide) invece di n per evitare size mismatch.
        """
        B = len(indices)

        n_children = self.n_children_mm[indices]
        max_n      = int(n_children.max()) if B > 0 else 1

        children_padded = torch.zeros(B, max_n, 13, 8, 8)
        child_mask      = torch.zeros(B, max_n, dtype=torch.bool)

        for i, (idx, n) in enumerate(zip(indices, n_children)):
            if n == 0:
                continue
            raw_fens = self.child_fens_mm[idx, :n]
            fens = [f.decode("ascii").rstrip("\x00").strip() for f in raw_fens]
            fens = [f for f in fens if f and ' ' in f]
            n_valid = len(fens)
            if n_valid == 0:
                continue
            children_padded[i, :n_valid] = encode_board_batch(fens)
            child_mask[i, :n_valid]      = True

        return children_padded.to(device), child_mask.to(device)


# ---------------------------------------------------------------------------
# Dataset — usa indice originale CSV per lookup child FEN
# ---------------------------------------------------------------------------

def eval_to_winprob(eval_str: str, turn: chess.Color) -> float:
    s = eval_str.strip()
    if s.startswith('#'):
        val_white = 1.0 if s.startswith('#+') else -1.0
    else:
        try:    cp = float(s)
        except: cp = 0.0
        cp = max(-2000.0, min(2000.0, cp))
        val_white = math.tanh(cp / CP_SCALE)
    return val_white if turn == chess.WHITE else -val_white


class StockfishDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        # Salva indici originali PRIMA del reset — corrispondono al memmap
        self.original_indices = df.index.tolist()
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
        value   = eval_to_winprob(eval_str, board.turn)

        target_vec = torch.zeros(len(legal_list))
        try:
            best  = chess.Move.from_uci(move_uci)
            idx_m = legal_list.index(best) if best in legal_list else 0
            target_vec[idx_m] = 1.0
        except:
            target_vec[0] = 1.0

        return {
            "board_t":  board_t,
            "moves_t":  moves_t,
            "policy_t": target_vec,
            "value_t":  torch.tensor([value], dtype=torch.float32),
            "n_moves":  len(legal_list),
            "ds_idx":   self.original_indices[idx],  # indice nella CSV originale
        }


def collate_fn(batch):
    max_n = max(item["n_moves"] for item in batch)
    B     = len(batch)

    boards_t      = torch.stack([item["board_t"] for item in batch])
    moves_padded  = torch.zeros(B, max_n, MOVE_VECTOR_DIM)
    move_mask     = torch.zeros(B, max_n, dtype=torch.bool)
    policy_padded = torch.zeros(B, max_n)
    values_t      = torch.stack([item["value_t"] for item in batch])
    ds_indices    = [item["ds_idx"] for item in batch]

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
        "ds_indices":   ds_indices,
    }


# ---------------------------------------------------------------------------
# Bellman loss
# ---------------------------------------------------------------------------

def bellman_consistency_loss(model, children_padded, child_mask, value_parent, tau):
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
def compute_value_std(v): return v.squeeze().float().std().item()

def bellman_tau(epoch):
    if epoch <= WARMUP_EPOCHS: return TAU_START
    p = (epoch - WARMUP_EPOCHS) / max(TOTAL_EPOCHS - WARMUP_EPOCHS, 1)
    return TAU_START + p * (TAU_FINAL - TAU_START)

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler":    scaler.state_dict()    if scaler    else None,
        "epoch":     epoch,
        "val_loss":  val_loss,
    }, tmp)
    os.replace(tmp, path)
    tqdm.write(f"  -> checkpoint: {path}  (epoch {epoch}, val_loss {val_loss:.4f})")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location=DEVICE)
    sd   = ckpt.get("model", ckpt)
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    if optimizer and "optimizer" in ckpt:
        try: optimizer.load_state_dict(ckpt["optimizer"])
        except: pass
    if scheduler and "scheduler" in ckpt and ckpt["scheduler"]:
        try: scheduler.load_state_dict(ckpt["scheduler"])
        except: pass
    if scaler and "scaler" in ckpt and ckpt["scaler"]:
        try: scaler.load_state_dict(ckpt["scaler"])
        except: pass
    epoch    = ckpt.get("epoch", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    tqdm.write(f"  -> caricato: {path}  (epoch {epoch})")
    return epoch, val_loss

# ---------------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, scaler, device, child_store,
              train=True, use_bellman=False, tau=TAU_START):

    model.train() if train else model.eval()
    for p in model.parameters(): p.requires_grad = train

    total_p = total_v = total_b = total_acc = 0.0
    vstd_list = []
    n_batches = bellman_skipped = 0

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for batch in tqdm(loader, leave=False):
            boards_t      = batch["boards_t"].to(device)
            moves_padded  = batch["moves_padded"].to(device)
            move_mask     = batch["move_mask"].to(device)
            policy_padded = batch["policy_padded"].to(device)
            values_t      = batch["values_t"].to(device)
            ds_indices    = batch["ds_indices"]

            with torch.amp.autocast("cuda", enabled=USE_AMP):
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
                    valid_local  = has_children.nonzero(as_tuple=True)[0]
                    if len(valid_local) > 0:
                        chosen = valid_local[
                            torch.randperm(len(valid_local), device=device)[:BELLMAN_SUBSAMPLE]
                        ].cpu().tolist()
                        global_indices = [ds_indices[i] for i in chosen]

                        children_padded, child_mask_b = child_store.get_children(
                            global_indices, device
                        )

                        with torch.amp.autocast("cuda", enabled=USE_AMP):
                            b_loss_val = bellman_consistency_loss(
                                model, children_padded, child_mask_b,
                                value_pred[chosen], tau,
                            )
                        loss = loss + BELLMAN_WEIGHT * b_loss_val

            if train:
                optimizer.zero_grad()
                if USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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
    print(f"Device: {DEVICE}  AMP: {USE_AMP}")
    print(f"Warmup: {WARMUP_EPOCHS} epoche  Bellman lambda={BELLMAN_WEIGHT}")
    print(f"tau: {TAU_START} -> {TAU_FINAL}\n")

    child_store = ChildFenStore(CHILD_FENS_DIR)

    # Reset index PRIMA dello split — garantisce corrispondenza con il memmap
    df = pd.read_csv(STOCKFISH_CSV)
    df = df.dropna(subset=["FEN", "Evaluation", "Move"])
    df = df[df["Evaluation"].astype(str).str.strip() != ""]
    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df = df.iloc[:MAX_SAMPLES]
    df = df.reset_index(drop=True)   # indici 0..N-1 = stesso ordine del precompute
    print(f"Posizioni: {len(df):,}")

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

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_val_loss = float("inf")
    start_epoch   = 1
    if os.path.exists(CHECKPOINT_OUT):
        start_epoch, best_val_loss = load_checkpoint(
            CHECKPOINT_OUT, model, optimizer, scheduler, scaler
        )
        start_epoch += 1

    print(f"Parametri: {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        use_bellman = epoch > WARMUP_EPOCHS
        tau         = bellman_tau(epoch)
        phase       = "warmup          " if not use_bellman else f"bellman tau={tau:.2f}"

        train_s = run_epoch(model, train_loader, optimizer, scaler, DEVICE,
                            child_store, train=True,
                            use_bellman=use_bellman, tau=tau)
        val_s   = run_epoch(model, val_loader, optimizer, scaler, DEVICE,
                            child_store, train=False)

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
            print(f"  ATTENZIONE collasso: {train_s['bellman_skipped']} batch skippati")
        gap = val_s["policy_loss"] - train_s["policy_loss"]
        if gap > 0.8:
            print(f"  ATTENZIONE overfitting: gap={gap:.3f}")

        save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                        val_s["policy_loss"], CHECKPOINT_OUT)
        if val_s["policy_loss"] < best_val_loss:
            best_val_loss = val_s["policy_loss"]
            save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                            best_val_loss, os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  * Nuovo best: {best_val_loss:.4f}")

    print(f"\nCompletato. Best val policy_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
