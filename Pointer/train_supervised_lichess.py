"""
train_supervised_lichess.py — Training supervisionato su partite Lichess filtrate.

Usa filtered_games.csv prodotto da filter_lichess.py.

Policy target: one-hot sulla mossa giocata dal giocatore (mossa umana 2000+ Elo)
Value target:  outcome Monte Carlo (+1/-1/0) dal punto di vista del giocatore che muove

Questo script:
  1. Carica il CSV e costruisce un dataset PyTorch
  2. Allena policy head + value head con backbone congelato per FREEZE_EPOCHS epoche
  3. Scongela il backbone e allena tutto insieme per le epoche rimanenti
  4. Salva checkpoint compatibile con il training RL (stessa struttura di train_alphazero_v2.py)

Utilizzo:
  python train_supervised_lichess.py

Dipendenze: torch, pandas, python-chess, tqdm
"""

import os
import math
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import chess

from MLChess import encode_board, encode_legal_moves, JellyFishPointer

MOVE_VECTOR_DIM = 46

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

FILTERED_CSV     = "filtered_games.csv"
CHECKPOINT_IN    = "checkpoints_az_v2/last.pt"   # parte da qui se esiste
CHECKPOINT_DIR   = "checkpoints_lichess"
CHECKPOINT_OUT   = os.path.join(CHECKPOINT_DIR, "last.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
TOTAL_EPOCHS   = 30
FREEZE_EPOCHS  = 5 #10          # prime N epoche: backbone congelato, solo heads
BATCH_SIZE     = 256
LR_BACKBONE    = 1e-5 #5e-6        # molto basso — il backbone ha già imparato qualcosa
LR_HEADS       = 5e-4
VALUE_LOSS_WEIGHT = 2.0      # più basso che nel RL perché il segnale MC è già coerente

# Dataset
MAX_SAMPLES    = 2_000_000   # campiona al massimo N posizioni dal CSV
VAL_FRACTION   = 0.02        # 2% per validation
NUM_WORKERS    = 4

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LichessDataset(Dataset):
    """
    Dataset di posizioni da partite Lichess filtrate.
    Ogni campione: (board_fen, move_uci, outcome)
    Encoding fatto on-the-fly per semplicità — se il dataset è grande
    considera pre-encoding su disco.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        fen      = row["fen"]
        move_uci = row["move_uci"]
        outcome  = float(row["outcome"])

        board = chess.Board(fen)
        legal_list = list(board.legal_moves)

        # Board tensor
        board_t = encode_board(fen)                  # (13, 8, 8)

        # Moves tensor
        moves_t = encode_legal_moves(board)          # (n_moves, 46)

        # Policy target: one-hot sulla mossa giocata
        target_vec = torch.zeros(len(legal_list))
        try:
            played = chess.Move.from_uci(move_uci)
            if played in legal_list:
                target_vec[legal_list.index(played)] = 1.0
            else:
                # Mossa non legale nella posizione (anomalia nel dataset)
                target_vec[0] = 1.0
        except Exception:
            target_vec[0] = 1.0

        return {
            "board_t":    board_t,                   # (13, 8, 8)
            "moves_t":    moves_t,                   # (n_moves, 46)
            "policy_t":   target_vec,                # (n_moves,)
            "value_t":    torch.tensor([outcome], dtype=torch.float32),  # (1,)
            "n_moves":    len(legal_list),
        }


def collate_fn(batch):
    """Padding delle mosse al massimo del batch."""
    max_n = max(item["n_moves"] for item in batch)
    B     = len(batch)

    boards_t      = torch.stack([item["board_t"] for item in batch])
    moves_padded  = torch.zeros(B, max_n, MOVE_VECTOR_DIM)
    move_mask     = torch.zeros(B, max_n, dtype=torch.bool)
    policy_padded = torch.zeros(B, max_n)
    values_t      = torch.stack([item["value_t"] for item in batch])

    for i, item in enumerate(batch):
        n = item["n_moves"]
        moves_padded[i, :n]  = item["moves_t"]
        move_mask[i, :n]     = True
        policy_padded[i, :n] = item["policy_t"]

    return {
        "boards_t":     boards_t,       # (B, 13, 8, 8)
        "moves_padded": moves_padded,   # (B, max_n, 46)
        "move_mask":    move_mask,      # (B, max_n)
        "policy_padded":policy_padded,  # (B, max_n)
        "values_t":     values_t,       # (B, 1)
    }


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
    tqdm.write(f"  → checkpoint salvato: {path}  (epoch {epoch}, val_loss {val_loss:.4f})")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=DEVICE)
    sd   = ckpt.get("model", ckpt)
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    if optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            pass
    if scheduler and "scheduler" in ckpt and ckpt["scheduler"]:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass
    epoch    = ckpt.get("epoch", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    tqdm.write(f"  → checkpoint caricato: {path}  (epoch {epoch})")
    return epoch, val_loss


# ---------------------------------------------------------------------------
# Training / Validation step
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, train=True, freeze_backbone=False):
    model.train() if train else model.eval()

    if freeze_backbone:
        # Congela backbone — solo heads ricevono gradiente
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.move_encoder.parameters():
            p.requires_grad = False
    else:
        for p in model.parameters():
            p.requires_grad = True

    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_accuracy    = 0.0
    n_batches         = 0

    ctx = torch.no_grad() if not train else torch.enable_grad()

    with ctx:
        for batch in tqdm(loader):
            boards_t      = batch["boards_t"].to(device)
            moves_padded  = batch["moves_padded"].to(device)
            move_mask     = batch["move_mask"].to(device)
            policy_padded = batch["policy_padded"].to(device)
            values_t      = batch["values_t"].to(device)

            _, probs, value_pred = model(boards_t, moves_padded, move_mask)

            # Policy loss — cross-entropy con one-hot
            log_probs   = torch.log(probs + 1e-8)
            policy_loss = -(policy_padded * log_probs).sum(dim=1).mean()

            # Value loss — MSE con outcome Monte Carlo
            value_loss = F.mse_loss(value_pred, values_t)

            loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Accuracy: la mossa con prob massima coincide con la mossa target?
            pred_idx   = probs.argmax(dim=1)
            target_idx = policy_padded.argmax(dim=1)
            accuracy   = (pred_idx == target_idx).float().mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_accuracy    += accuracy
            n_batches         += 1

    return {
        "policy_loss": total_policy_loss / max(n_batches, 1),
        "value_loss":  total_value_loss  / max(n_batches, 1),
        "accuracy":    total_accuracy    / max(n_batches, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    # ---- Carica CSV ----
    print(f"Caricamento dataset: {FILTERED_CSV}")
    df = pd.read_csv(FILTERED_CSV)
    print(f"  Posizioni totali nel CSV: {len(df):,}")

    # Campiona se troppo grande
    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42)
        print(f"  Campionamento a {MAX_SAMPLES:,} posizioni")

    # Rimuovi posizioni con anomalie
    df = df.dropna(subset=["fen", "move_uci", "outcome"])
    df = df[df["outcome"].isin([1.0, 0.0, -1.0])]
    print(f"  Posizioni dopo pulizia: {len(df):,}")

    # Split train/val
    val_size  = max(1000, int(len(df) * VAL_FRACTION))
    val_df    = df.sample(n=val_size, random_state=42)
    train_df  = df.drop(val_df.index)
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}")
    print(f"  Distribuzione outcome — +1: {(df['outcome']==1.0).sum():,}  "
          f"0: {(df['outcome']==0.0).sum():,}  "
          f"-1: {(df['outcome']==-1.0).sum():,}")

    # ---- Dataset e DataLoader ----
    train_ds = LichessDataset(train_df)
    val_ds   = LichessDataset(val_df)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ---- Modello ----
    model = JellyFishPointer().to(DEVICE)

    if os.path.exists(CHECKPOINT_IN):
        print(f"Carico checkpoint di partenza: {CHECKPOINT_IN}")
        ckpt = torch.load(CHECKPOINT_IN, map_location=DEVICE)
        sd   = ckpt.get("model", ckpt)
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
    else:
        print("Nessun checkpoint trovato, parto da zero.")

    optimizer = Adam([
        {"params": list(model.backbone.parameters()) +
                   list(model.move_encoder.parameters()),
         "lr": LR_BACKBONE},
        {"params": list(model.policy_head.parameters()) +
                   list(model.value_head.parameters()),
         "lr": LR_HEADS},
    ])

    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)

    best_val_loss = float("inf")
    start_epoch   = 1

    # Carica checkpoint lichess se esiste (ripresa training)
    if os.path.exists(CHECKPOINT_OUT):
        start_epoch, best_val_loss = load_checkpoint(
            CHECKPOINT_OUT, model, optimizer, scheduler
        )
        start_epoch += 1

    print(f"\nParametri: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Inizio training per {TOTAL_EPOCHS} epoche (da {start_epoch})")
    print(f"Fase 1: backbone congelato per le prime {FREEZE_EPOCHS} epoche")
    print(f"Fase 2: fine-tuning completo dopo l'epoca {FREEZE_EPOCHS}\n")

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        freeze = epoch <= FREEZE_EPOCHS

        if freeze and epoch == start_epoch:
            print(f"[Fase 1] Backbone congelato — LR heads: {LR_HEADS}")
        elif not freeze and (epoch == FREEZE_EPOCHS + 1 or epoch == start_epoch):
            print(f"[Fase 2] Fine-tuning completo — LR backbone: {LR_BACKBONE}, LR heads: {LR_HEADS}")

        # Training
        train_stats = run_epoch(
            model, train_loader, optimizer, DEVICE,
            train=True, freeze_backbone=freeze,
        )

        # Validation
        val_stats = run_epoch(
            model, val_loader, optimizer, DEVICE,
            train=False, freeze_backbone=False,
        )

        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{TOTAL_EPOCHS} | "
            f"{'[frozen] ' if freeze else '[full]   '}"
            f"train p_loss: {train_stats['policy_loss']:.4f}  "
            f"v_loss: {train_stats['value_loss']:.4f}  "
            f"acc: {train_stats['accuracy']:.3f}  |  "
            f"val p_loss: {val_stats['policy_loss']:.4f}  "
            f"v_loss: {val_stats['value_loss']:.4f}  "
            f"acc: {val_stats['accuracy']:.3f}"
        )

        # Salva last
        save_checkpoint(model, optimizer, scheduler, epoch, val_stats["policy_loss"], CHECKPOINT_OUT)

        # Salva best
        if val_stats["policy_loss"] < best_val_loss:
            best_val_loss = val_stats["policy_loss"]
            best_path     = os.path.join(CHECKPOINT_DIR, "best.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, best_path)
            print(f"  ★ Nuovo best val policy_loss: {best_val_loss:.4f}")

    print(f"\nTraining completato. Best val policy_loss: {best_val_loss:.4f}")
    print(f"Checkpoint finale: {CHECKPOINT_OUT}")
    print(f"\nPer riprendere il training RL, imposta CHECKPOINT_IN = '{CHECKPOINT_OUT}'")
    print("in train_alphazero_v2.py")


if __name__ == "__main__":
    main()
