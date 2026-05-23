
"""
train_value_head.py — Riallena solo la value head su chessData.csv.

Congela backbone e policy head, allena solo value head con MSE loss
su posizioni e valutazioni Stockfish.

Carica dal checkpoint RL last, salva in checkpoints_value/.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from MLChess import encode_board, JellyFishPointer

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

CSV_FILE          = "chessData.csv"
RL_CHECKPOINT     = "checkpoints_az/last.pt"
VALUE_CHECKPOINT  = "checkpoints_value"

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS      = 20
BATCH_SIZE  = 512
LR          = 1e-3
NUM_WORKERS = 4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def encode_value(evaluation: str, max_cp: int = 1000) -> float:
    if '#' in evaluation:
        return 1.0 if '+' in evaluation or not evaluation.startswith('-') else -1.0
    try:
        cp = max(-max_cp, min(max_cp, int(evaluation)))
        return cp / max_cp
    except ValueError:
        return 0.0


class ValueDataset(Dataset):
    def __init__(self, csv_file: str, split: str = 'train'):
        df = pd.read_csv(csv_file)
        # Rimuovi righe con valutazioni mancanti
        df = df.dropna(subset=['Evaluation']).reset_index(drop=True)

        n = len(df)
        train_end = int(n * 0.85)

        if split == 'train':
            self.df = df.iloc[:train_end].reset_index(drop=True)
        else:
            self.df = df.iloc[train_end:].reset_index(drop=True)

        print(f"  {split}: {len(self.df)} posizioni")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        board = encode_board(str(row['FEN']))
        value = encode_value(str(row['Evaluation']))
        return board, torch.tensor(value, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    tqdm.write(f"  → checkpoint salvato: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print(f"Caricamento dataset da {CSV_FILE}...")

    trainset = ValueDataset(CSV_FILE, split='train')
    valset   = ValueDataset(CSV_FILE, split='val')

    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda')
    )
    valloader = DataLoader(
        valset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda')
    )

    # Carica modello
    model = JellyFishPointer().to(DEVICE)
    ckpt  = torch.load(RL_CHECKPOINT, map_location=DEVICE)
    state_dict = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"Checkpoint caricato: {RL_CHECKPOINT}")

    # Congela tutto tranne value head
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.move_encoder.parameters():
        p.requires_grad = False
    for p in model.policy_head.parameters():
        p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri trainabili (solo value head): {trainable:,}\n")

    optimizer = Adam(model.value_head.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    os.makedirs(VALUE_CHECKPOINT, exist_ok=True)

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Epoche", dynamic_ncols=True)

    for epoch in epoch_bar:

        # Training
        model.train()
        total_train_loss = 0.0

        train_bar = tqdm(trainloader, desc=f"  Epoch {epoch:03d} [train]", leave=False, dynamic_ncols=True)
        for boards, values in train_bar:
            boards = boards.to(DEVICE)
            values = values.to(DEVICE).unsqueeze(1)

            # Forward solo backbone + value head
            with torch.no_grad():
                h = model.backbone(boards)
            value_pred = model.value_head(h)

            loss = criterion(value_pred, values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * boards.size(0)
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(trainset)

        # Validation
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for boards, values in tqdm(valloader, desc=f"  Epoch {epoch:03d} [val]  ", leave=False, dynamic_ncols=True):
                boards = boards.to(DEVICE)
                values = values.to(DEVICE).unsqueeze(1)
                h      = model.backbone(boards)
                value_pred = model.value_head(h)
                loss   = criterion(value_pred, values)
                total_val_loss += loss.item() * boards.size(0)

        avg_val_loss = total_val_loss / len(valset)
        scheduler.step(avg_val_loss)

        tqdm.write(
            f"Epoch {epoch:03d} | "
            f"train_loss: {avg_train_loss:.4f}  "
            f"val_loss: {avg_val_loss:.4f}  "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        checkpoint_state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss":  avg_val_loss,
        }

        save_checkpoint(checkpoint_state, os.path.join(VALUE_CHECKPOINT, "last.pt"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(checkpoint_state, os.path.join(VALUE_CHECKPOINT, "best.pt"))
            tqdm.write(f"  ★ Nuovo best val loss: {best_val_loss:.4f}")

        epoch_bar.set_postfix({
            "val_loss": f"{avg_val_loss:.4f}",
            "best":     f"{best_val_loss:.4f}",
        })

    print(f"\nTraining completato. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint salvato in {VALUE_CHECKPOINT}/best.pt")
    print(f"\nOra riprendi il RL con SUPERVISED_CHECKPOINT = '{VALUE_CHECKPOINT}/best.pt'")


if __name__ == "__main__":
    main()
