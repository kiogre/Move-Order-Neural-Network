"""
train_supervised_gcn.py — Training supervisionato per JellyFishPointerGCN.

Usa filtered_games.csv (colonne: fen, move_uci, outcome).

Board codificata come grafo PyG (64 nodi, edge fissi regina+cavallo).
Policy target: one-hot sulla mossa giocata.
Value target:  outcome Monte Carlo (+1/-1/0) dal punto di vista del giocatore che muove.

Utilizzo:
  python train_supervised_gcn.py

Dipendenze: torch, torch_geometric, pandas, python-chess, tqdm
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch
from tqdm import tqdm
import chess

from MLChess import JellyFishPointerGCN, encode_board_graph, encode_legal_moves, encode_result

MOVE_VECTOR_DIM = 46

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

FILTERED_CSV    = "filtered_games.csv"
CHECKPOINT_IN   = "checkpoints_gcn/last.pt"                              # es. "checkpoints_gcn/last.pt"
CHECKPOINT_DIR  = "checkpoints_gcn"
CHECKPOINT_OUT  = os.path.join(CHECKPOINT_DIR, "last.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOTAL_EPOCHS      = 30
BATCH_SIZE        = 256
LR                = 1e-3
VALUE_LOSS_WEIGHT = 2.0
MAX_SAMPLES       = 2_000_000
VAL_FRACTION      = 0.02
NUM_WORKERS       = 4

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LichessGraphDataset(Dataset):
    """
    Identico a LichessDataset ma con board → grafo PyG invece di (13,8,8).
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

        # Board → grafo PyG  (x: 64×15, edge_index fisso)
        graph = encode_board_graph(fen)

        # Global features (turno, arrocco, en passant, fullmove)
        gf = []
        gf.append(1.0 if board.turn == chess.WHITE else -1.0)
        gf.append(1.0 if board.has_kingside_castling_rights(chess.WHITE)  else 0.0)
        gf.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
        gf.append(1.0 if board.has_kingside_castling_rights(chess.BLACK)  else 0.0)
        gf.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
        gf.append(1.0 if board.ep_square is not None else 0.0)
        gf.append(board.fullmove_number / 100.0)
        graph.global_features = torch.tensor(gf, dtype=torch.float32)

        # Mosse legali  (N, 46)
        moves_t = encode_legal_moves(board)

        # Policy target: one-hot sulla mossa giocata
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
            "graph":    graph,
            "moves_t":  moves_t,                                             # (N, 46)
            "policy_t": target_vec,                                          # (N,)
            "value_t":  torch.tensor([outcome], dtype=torch.float32),        # (1,)
            "n_moves":  len(legal_list),
        }


def collate_fn(batch):
    """
    - Grafi PyG → Batch.from_data_list  (batch.x, batch.edge_index, batch.batch)
    - Mosse e policy → padding al massimo N del batch
    """
    max_n = max(item["n_moves"] for item in batch)
    B     = len(batch)

    # Batch PyG: concatena grafi (nodi e archi tutti insieme + batch vector)
    graph_batch   = Batch.from_data_list([item["graph"] for item in batch])

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
        "graph_batch":   graph_batch,     # PyG Batch
        "moves_padded":  moves_padded,    # (B, max_n, 46)
        "move_mask":     move_mask,       # (B, max_n)
        "policy_padded": policy_padded,   # (B, max_n)
        "values_t":      values_t,        # (B, 1)
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
        try:   optimizer.load_state_dict(ckpt["optimizer"])
        except Exception: pass
    if scheduler and "scheduler" in ckpt and ckpt["scheduler"]:
        try:   scheduler.load_state_dict(ckpt["scheduler"])
        except Exception: pass
    epoch    = ckpt.get("epoch", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    tqdm.write(f"  → checkpoint caricato: {path}  (epoch {epoch})")
    return epoch, val_loss


# ---------------------------------------------------------------------------
# Training / Validation step
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_accuracy    = 0.0
    n_batches         = 0

    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in tqdm(loader, leave=False):
            graph_batch   = batch["graph_batch"].to(device)
            moves_padded  = batch["moves_padded"].to(device)
            move_mask     = batch["move_mask"].to(device)
            policy_padded = batch["policy_padded"].to(device)
            values_t      = batch["values_t"].to(device)

            _, probs, value_pred = model(graph_batch, moves_padded, move_mask)

            # Policy loss — cross-entropy con distribuzione one-hot
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
    print(f"  Posizioni totali: {len(df):,}")

    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42)
        print(f"  Campionamento a {MAX_SAMPLES:,} posizioni")

    df = df.dropna(subset=["fen", "move_uci", "outcome"])
    df = df[df["outcome"].isin([1.0, 0.0, -1.0])]
    print(f"  Dopo pulizia: {len(df):,}")

    val_size = max(1000, int(len(df) * VAL_FRACTION))
    val_df   = df.sample(n=val_size, random_state=42)
    train_df = df.drop(val_df.index)
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}")
    print(f"  Outcome — +1: {(df['outcome']==1.0).sum():,}  "
          f"0: {(df['outcome']==0.0).sum():,}  "
          f"-1: {(df['outcome']==-1.0).sum():,}")

    # ---- DataLoader ----
    train_ds = LichessGraphDataset(train_df)
    val_ds   = LichessGraphDataset(val_df)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ---- Modello ----
    model = JellyFishPointerGCN().to(DEVICE)
    print(f"Parametri: {sum(p.numel() for p in model.parameters()):,}")

    if CHECKPOINT_IN and os.path.exists(CHECKPOINT_IN):
        print(f"Carico checkpoint: {CHECKPOINT_IN}")
        ckpt = torch.load(CHECKPOINT_IN, map_location=DEVICE)
        sd   = ckpt.get("model", ckpt)
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
    else:
        print("Nessun checkpoint, parto da zero.")

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)

    best_val_loss = float("inf")
    start_epoch   = 1

    if os.path.exists(CHECKPOINT_OUT):
        start_epoch, best_val_loss = load_checkpoint(
            CHECKPOINT_OUT, model, optimizer, scheduler
        )
        start_epoch += 1

    print(f"\nTraining per {TOTAL_EPOCHS} epoche (da {start_epoch})\n")

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):

        train_stats = run_epoch(model, train_loader, optimizer, DEVICE, train=True)
        val_stats   = run_epoch(model, val_loader,   optimizer, DEVICE, train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{TOTAL_EPOCHS} | "
            f"train  p_loss: {train_stats['policy_loss']:.4f}  "
            f"v_loss: {train_stats['value_loss']:.4f}  "
            f"acc: {train_stats['accuracy']:.3f}  |  "
            f"val  p_loss: {val_stats['policy_loss']:.4f}  "
            f"v_loss: {val_stats['value_loss']:.4f}  "
            f"acc: {val_stats['accuracy']:.3f}"
        )

        save_checkpoint(model, optimizer, scheduler, epoch,
                        val_stats["policy_loss"], CHECKPOINT_OUT)

        if val_stats["policy_loss"] < best_val_loss:
            best_val_loss = val_stats["policy_loss"]
            best_path     = os.path.join(CHECKPOINT_DIR, "best.pt")
            save_checkpoint(model, optimizer, scheduler, epoch,
                            best_val_loss, best_path)
            print(f"  ★ Nuovo best val policy_loss: {best_val_loss:.4f}")

    print(f"\nTraining completato. Best val policy_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
