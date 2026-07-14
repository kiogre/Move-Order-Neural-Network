"""
train_supervised_bellman.py — Training supervisionato + Bellman consistency loss.

Estende train_supervised_lichess.py aggiungendo un termine ausiliario:

    L_bellman = ( V(s) + softmin_{s'} V(s') )^2

dove softmin_tau(x) = -tau * logsumexp(-x / tau)   [min differenziabile]

Logica:
  - V(s) è la win-prob dal punto di vista di chi muove
  - Dopo la mossa il turno passa all'avversario
  - V(s') è la win-prob dell'avversario nella posizione figlia
  - L'ottimalità minimax richiede: V(s) = -min_{s'} V(s')
    ossia V(s) + min_{s'} V(s') = 0

Note implementative:
  - Il branching medio è ~35: per batch=256 si avrebbero ~8960 board figli
    da passare nel backbone, troppi per una 3060 con gradiente.
    Soluzione: subsample BELLMAN_SUBSAMPLE posizioni per batch.
  - Le posizioni terminali (0 mosse legali) vengono escluse automaticamente.
  - Il peso BELLMAN_WEIGHT va tenuto piccolo (0.05-0.2): è un termine
    ausiliario, non deve competere con la loss principale.
  - Monitora value_calib / output range del value head: se collassa
    (range < 0.05) abbassa BELLMAN_WEIGHT o disabilita temporaneamente.

Utilizzo:
  python train_supervised_bellman.py

Dipendenze: torch, pandas, python-chess, tqdm
"""

import os
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

FILTERED_CSV    = "filtered_games.csv"
CHECKPOINT_IN   = "checkpoints_lichess/last.pt"
CHECKPOINT_DIR  = "checkpoints_bellman"
CHECKPOINT_OUT  = os.path.join(CHECKPOINT_DIR, "last.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
TOTAL_EPOCHS      = 30
FREEZE_EPOCHS     = 0
BATCH_SIZE        = 256
LR_BACKBONE       = 1e-3
LR_HEADS          = 1e-3
VALUE_LOSS_WEIGHT = 2.0

# Bellman auxiliary loss
BELLMAN_WEIGHT     = 0.1    # λ: peso del termine ausiliario. Inizia piccolo.
BELLMAN_TAU        = 0.5    # temperatura softmin: alto=più smooth, basso=più vicino al vero min
BELLMAN_SUBSAMPLE  = 64     # quante posizioni del batch usare per Bellman (memoria)
                             # Con ~35 figli ciascuna → ~2240 board extra nel backbone

# Dataset
MAX_SAMPLES    = 2_000_000
VAL_FRACTION   = 0.02
NUM_WORKERS    = 4

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LichessDataset(Dataset):
    """
    Dataset identico all'originale, con l'aggiunta del FEN grezzo nel sample.
    Il FEN serve per generare i board figli on-demand solo per le posizioni
    subsamplate nel Bellman loss — non qui, dove sarebbe pagato per tutti.
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

        board      = chess.Board(fen)
        legal_list = list(board.legal_moves)

        board_t = encode_board(fen)               # (13, 8, 8)
        moves_t = encode_legal_moves(board)       # (n_moves, 46)

        target_vec = torch.zeros(len(legal_list))
        try:
            played = chess.Move.from_uci(move_uci)
            if played in legal_list:
                target_vec[legal_list.index(played)] = 1.0
            else:
                target_vec[0] = 1.0
        except Exception:
            target_vec[0] = 1.0

        return {
            "board_t":  board_t,
            "moves_t":  moves_t,
            "policy_t": target_vec,
            "value_t":  torch.tensor([outcome], dtype=torch.float32),
            "n_moves":  len(legal_list),
            "fen":      fen,          # ← usato solo dal Bellman subsample
        }


def collate_fn(batch):
    """Padding identico all'originale. I FEN vengono passati come lista."""
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
        "boards_t":     boards_t,       # (B, 13, 8, 8)
        "moves_padded": moves_padded,   # (B, max_n, 46)
        "move_mask":    move_mask,      # (B, max_n) bool
        "policy_padded":policy_padded,  # (B, max_n)
        "values_t":     values_t,       # (B, 1)
        "fens":         fens,           # list[str], len=B
    }


def build_children_batch(fens: list[str], device: torch.device):
    """
    Dato un sottoinsieme di FEN, genera i board figli e le mask.
    Chiamato solo per BELLMAN_SUBSAMPLE posizioni per batch — non per tutto il batch.

    Returns:
        children_padded : (B_sub, max_n, 13, 8, 8)  su device
        child_mask      : (B_sub, max_n) bool         su device
    """
    all_children = []
    all_masks    = []
    max_n        = 0

    for fen in fens:
        board      = chess.Board(fen)
        legal_list = list(board.legal_moves)
        child_boards = []
        for move in legal_list:
            board.push(move)
            child_boards.append(encode_board(board.fen()))
            board.pop()
        all_children.append(child_boards)
        all_masks.append(len(child_boards))
        max_n = max(max_n, len(child_boards))

    B_sub           = len(fens)
    children_padded = torch.zeros(B_sub, max_n, 13, 8, 8)
    child_mask      = torch.zeros(B_sub, max_n, dtype=torch.bool)

    for i, (child_boards, n) in enumerate(zip(all_children, all_masks)):
        if n > 0:
            children_padded[i, :n] = torch.stack(child_boards)
            child_mask[i, :n]      = True

    return children_padded.to(device), child_mask.to(device)


# ---------------------------------------------------------------------------
# Bellman consistency loss
# ---------------------------------------------------------------------------

def bellman_consistency_loss(
    model:           nn.Module,
    children_padded: torch.Tensor,   # (B_sub, max_n, 13, 8, 8)
    move_mask:       torch.Tensor,   # (B_sub, max_n) bool
    value_parent:    torch.Tensor,   # (B_sub, 1)
    tau:             float = 0.5,
) -> torch.Tensor:
    """
    Calcola il termine ausiliario Bellman:

        L = mean_s ( V(s) + softmin_{s'} V(s') )^2

    dove softmin_tau(X) = -tau * logsumexp(-X / tau)

    I board figli vengono passati nel backbone+value_head in un'unica
    forward pass flattenata (B_sub * max_n, 13, 8, 8), con le posizioni
    di padding mascherate prima del logsumexp.

    Args:
        model:           JellyFishPointer (train mode, gradiente attivo)
        children_padded: board figli paddati
        move_mask:       True per slot reali, False per padding
        value_parent:    V(s) già calcolato nel forward pass principale
        tau:             temperatura del softmin

    Returns:
        loss scalare
    """
    B_sub, max_n, C, H, W = children_padded.shape

    # Flatten: (B_sub * max_n, 13, 8, 8)
    children_flat = children_padded.view(B_sub * max_n, C, H, W)

    # Forward backbone + value_head sui figli.
    # Usiamo solo queste due componenti — la policy dei figli non serve qui.
    h_children = model.backbone(children_flat)               # (B_sub*max_n, 512)
    v_children = model.value_head(h_children)                # (B_sub*max_n, 1)
    v_children = v_children.view(B_sub, max_n)               # (B_sub, max_n)

    # Maschera il padding: slot non reali → +inf prima di logsumexp(-x/tau)
    # in modo che exp(-inf/tau) = 0 e non contribuiscano al softmin
    INF = torch.finfo(v_children.dtype).max / 2
    v_children_masked = v_children.masked_fill(~move_mask, INF)

    # softmin differenziabile: -tau * logsumexp(-x / tau)
    # = -tau * log( sum_i exp(-x_i / tau) )   [solo sugli slot reali]
    soft_min = -tau * torch.logsumexp(-v_children_masked / tau, dim=1)  # (B_sub,)

    # Vincolo: V(s) + softmin_{s'} V(s') = 0
    v_parent = value_parent.squeeze(1)                       # (B_sub,)
    residual = v_parent + soft_min                           # (B_sub,)

    return (residual ** 2).mean()


# ---------------------------------------------------------------------------
# Diagnostica value head
# ---------------------------------------------------------------------------

@torch.no_grad()
def value_diagnostics(value_pred: torch.Tensor) -> dict:
    """
    Monitora il range di output del value head.
    Se std < 0.05 o range < 0.1, il value head sta collassando.
    """
    v = value_pred.squeeze().float()
    return {
        "v_mean": v.mean().item(),
        "v_std":  v.std().item(),
        "v_min":  v.min().item(),
        "v_max":  v.max().item(),
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
        try:    optimizer.load_state_dict(ckpt["optimizer"])
        except Exception: pass
    if scheduler and "scheduler" in ckpt and ckpt["scheduler"]:
        try:    scheduler.load_state_dict(ckpt["scheduler"])
        except Exception: pass
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
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.move_encoder.parameters():
            p.requires_grad = False
    else:
        for p in model.parameters():
            p.requires_grad = True

    total_policy_loss   = 0.0
    total_value_loss    = 0.0
    total_bellman_loss  = 0.0
    total_accuracy      = 0.0
    n_batches           = 0

    ctx = torch.no_grad() if not train else torch.enable_grad()

    with ctx:
        for batch in tqdm(loader, leave=False):
            boards_t      = batch["boards_t"].to(device)
            moves_padded  = batch["moves_padded"].to(device)
            move_mask     = batch["move_mask"].to(device)
            policy_padded = batch["policy_padded"].to(device)
            values_t      = batch["values_t"].to(device)
            fens          = batch["fens"]

            B = boards_t.shape[0]

            # --- Forward pass principale ---
            _, probs, value_pred = model(boards_t, moves_padded, move_mask)

            # Policy loss: cross-entropy con one-hot
            log_probs   = torch.log(probs + 1e-8)
            policy_loss = -(policy_padded * log_probs).sum(dim=1).mean()

            # Value loss: MSE con outcome Monte Carlo
            value_loss = F.mse_loss(value_pred, values_t)

            loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

            # --- Bellman consistency loss (solo in training) ---
            # I figli vengono generati QUI, solo per le posizioni subsamplate.
            # Costo: BELLMAN_SUBSAMPLE * ~35 encode_board per batch, non B * 35.
            bellman_loss_val = torch.tensor(0.0, device=device)
            if train and BELLMAN_WEIGHT > 0:
                # Subsample casuale di indici validi (con almeno 1 mossa legale)
                has_children = move_mask.any(dim=1)
                valid_idx    = has_children.nonzero(as_tuple=True)[0]

                if len(valid_idx) > 0:
                    sub_idx = valid_idx[
                        torch.randperm(len(valid_idx), device=device)[:BELLMAN_SUBSAMPLE]
                    ]
                    sub_fens = [fens[i] for i in sub_idx.cpu().tolist()]

                    # Genera board figli solo per le posizioni subsamplate (CPU, veloce)
                    children_padded, child_mask = build_children_batch(sub_fens, device)

                    bellman_loss_val = bellman_consistency_loss(
                        model           = model,
                        children_padded = children_padded,
                        move_mask       = child_mask,
                        value_parent    = value_pred[sub_idx].detach(),
                        tau             = BELLMAN_TAU,
                    )

                    loss = loss + BELLMAN_WEIGHT * bellman_loss_val

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Accuracy
            pred_idx   = probs.argmax(dim=1)
            target_idx = policy_padded.argmax(dim=1)
            accuracy   = (pred_idx == target_idx).float().mean().item()

            total_policy_loss  += policy_loss.item()
            total_value_loss   += value_loss.item()
            total_bellman_loss += bellman_loss_val.item()
            total_accuracy     += accuracy
            n_batches          += 1

    denom = max(n_batches, 1)
    return {
        "policy_loss":  total_policy_loss  / denom,
        "value_loss":   total_value_loss   / denom,
        "bellman_loss": total_bellman_loss / denom,
        "accuracy":     total_accuracy     / denom,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print(f"Bellman weight: {BELLMAN_WEIGHT}  tau: {BELLMAN_TAU}  subsample: {BELLMAN_SUBSAMPLE}")

    print(f"Caricamento dataset: {FILTERED_CSV}")
    df = pd.read_csv(FILTERED_CSV)
    print(f"  Posizioni totali: {len(df):,}")

    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42)

    df = df.dropna(subset=["fen", "move_uci", "outcome"])
    df = df[df["outcome"].isin([1.0, 0.0, -1.0])]
    print(f"  Posizioni dopo pulizia: {len(df):,}")

    val_size = max(1000, int(len(df) * VAL_FRACTION))
    val_df   = df.sample(n=val_size, random_state=42)
    train_df = df.drop(val_df.index)
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}")

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

    model = JellyFishPointer().to(DEVICE)

    if os.path.exists(CHECKPOINT_IN):
        print(f"Carico checkpoint: {CHECKPOINT_IN}")
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

    scheduler     = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)
    best_val_loss = float("inf")
    start_epoch   = 1

    if os.path.exists(CHECKPOINT_OUT):
        start_epoch, best_val_loss = load_checkpoint(
            CHECKPOINT_OUT, model, optimizer, scheduler
        )
        start_epoch += 1

    print(f"\nParametri: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training per {TOTAL_EPOCHS} epoche (da {start_epoch})\n")

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        freeze = epoch <= FREEZE_EPOCHS

        train_stats = run_epoch(
            model, train_loader, optimizer, DEVICE,
            train=True, freeze_backbone=freeze,
        )
        val_stats = run_epoch(
            model, val_loader, optimizer, DEVICE,
            train=False, freeze_backbone=False,
        )

        scheduler.step()

        # Diagnostica value head sull'ultimo batch di val
        # (già calcolata implicitamente; qui solo stampa summary)
        print(
            f"Epoch {epoch:03d}/{TOTAL_EPOCHS} | "
            f"{'[frozen] ' if freeze else '[full]   '}"
            f"train  p: {train_stats['policy_loss']:.4f}  "
            f"v: {train_stats['value_loss']:.4f}  "
            f"b: {train_stats['bellman_loss']:.4f}  "
            f"acc: {train_stats['accuracy']:.3f}  |  "
            f"val  p: {val_stats['policy_loss']:.4f}  "
            f"v: {val_stats['value_loss']:.4f}  "
            f"acc: {val_stats['accuracy']:.3f}"
        )

        # Avviso collasso value head
        # (aggiungi qui value_calib o range check se hai già quella funzione)
        if train_stats["value_loss"] < 0.001:
            print("  ⚠️  value_loss quasi zero — possibile collasso del value head!")
            print("      Controlla il range di output. Considera di abbassare BELLMAN_WEIGHT.")

        save_checkpoint(model, optimizer, scheduler, epoch,
                        val_stats["policy_loss"], CHECKPOINT_OUT)

        if val_stats["policy_loss"] < best_val_loss:
            best_val_loss = val_stats["policy_loss"]
            save_checkpoint(model, optimizer, scheduler, epoch,
                            best_val_loss,
                            os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  ★ Nuovo best: {best_val_loss:.4f}")

    print(f"\nTraining completato. Best val policy_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
