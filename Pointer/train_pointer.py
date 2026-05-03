import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Adatta questi import ai tuoi nomi file
from MLChess import create_dataloaders_pointer
from MLChess import JellyFishPointer


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

CSV_FILE      = "../over_mate_1_tactic_evals.csv"
BATCH_SIZE    = 256 #128
NUM_WORKERS   = 4
LR            = 1e-3
EPOCHS        = 50
POLICY_WEIGHT = 1.0   # peso della policy loss nella loss totale
VALUE_WEIGHT  = 1.0   # peso della value loss nella loss totale
CHECKPOINT_DIR = "checkpoints_pointer"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Funzioni di training e validazione
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, policy_criterion, value_criterion, epoch, device):
    model.train()

    total_loss   = 0.0
    policy_loss_sum = 0.0
    value_loss_sum  = 0.0
    correct      = 0
    total        = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False, dynamic_ncols=True)

    for boards, moves, mask, labels, values in pbar:
        boards = boards.to(device)
        moves  = moves.to(device)
        mask   = mask.to(device)
        labels = labels.to(device)
        values = values.to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits, probs, value_pred = model(boards, moves, mask)

        p_loss = policy_criterion(logits, labels)
        v_loss = value_criterion(value_pred, values)
        loss   = POLICY_WEIGHT * p_loss + VALUE_WEIGHT * v_loss

        loss.backward()
        optimizer.step()

        # Statistiche
        batch_size     = boards.size(0)
        total_loss     += loss.item()    * batch_size
        policy_loss_sum += p_loss.item() * batch_size
        value_loss_sum  += v_loss.item() * batch_size

        preds    = logits.argmax(dim=1)
        correct  += (preds == labels).sum().item()
        total    += batch_size

        pbar.set_postfix({
            "loss":   f"{loss.item():.4f}",
            "p_loss": f"{p_loss.item():.4f}",
            "v_loss": f"{v_loss.item():.4f}",
            "acc":    f"{correct/total:.3f}",
        })

    n = len(loader.dataset)
    return {
        "loss":         total_loss    / n,
        "policy_loss":  policy_loss_sum / n,
        "value_loss":   value_loss_sum  / n,
        "accuracy":     correct / total,
    }


@torch.no_grad()
def validate(model, loader, policy_criterion, value_criterion, epoch, device):
    model.eval()

    total_loss      = 0.0
    policy_loss_sum = 0.0
    value_loss_sum  = 0.0
    correct         = 0
    total           = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False, dynamic_ncols=True)

    for boards, moves, mask, labels, values in pbar:
        boards = boards.to(device)
        moves  = moves.to(device)
        mask   = mask.to(device)
        labels = labels.to(device)
        values = values.to(device).unsqueeze(1)

        logits, probs, value_pred = model(boards, moves, mask)

        p_loss = policy_criterion(logits, labels)
        v_loss = value_criterion(value_pred, values)
        loss   = POLICY_WEIGHT * p_loss + VALUE_WEIGHT * v_loss

        batch_size      = boards.size(0)
        total_loss      += loss.item()    * batch_size
        policy_loss_sum += p_loss.item() * batch_size
        value_loss_sum  += v_loss.item() * batch_size

        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += batch_size

        pbar.set_postfix({
            "loss":   f"{loss.item():.4f}",
            "acc":    f"{correct/total:.3f}",
        })

    n = len(loader.dataset)
    return {
        "loss":        total_loss     / n,
        "policy_loss": policy_loss_sum / n,
        "value_loss":  value_loss_sum  / n,
        "accuracy":    correct / total,
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  → checkpoint salvato: {path}")


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"  → checkpoint caricato: {path}  (epoch {ckpt['epoch']}, best val loss {ckpt['best_val_loss']:.4f})")
    return ckpt["epoch"], ckpt["best_val_loss"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print(f"Caricamento dataset da {CSV_FILE}...")

    trainloader, valloader, _ = create_dataloaders_pointer(
        csv_file=CSV_FILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = JellyFishPointer().to(DEVICE)
    print(f"Parametri totali: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion  = nn.MSELoss()

    start_epoch    = 1
    best_val_loss  = float('inf')

    # Resume se esiste un checkpoint
    last_ckpt = os.path.join(CHECKPOINT_DIR, "last.pt")
    if os.path.exists(last_ckpt):
        print("Checkpoint trovato, riprendo il training...")
        start_epoch, best_val_loss = load_checkpoint(last_ckpt, model, optimizer, scheduler)
        start_epoch += 1

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------

    print(f"\nInizio training per {EPOCHS} epoche totali (da epoca {start_epoch})\n")

    epoch_bar = tqdm(range(start_epoch, EPOCHS + 1), desc="Epoche", dynamic_ncols=True)

    for epoch in epoch_bar:

        train_stats = train_one_epoch(
            model, trainloader, optimizer, policy_criterion, value_criterion, epoch, DEVICE
        )
        val_stats = validate(
            model, valloader, policy_criterion, value_criterion, epoch, DEVICE
        )

        scheduler.step(val_stats["loss"])

        # Stampa riepilogo epoca
        tqdm.write(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_stats['loss']:.4f}  acc: {train_stats['accuracy']:.3f}  "
            f"(p: {train_stats['policy_loss']:.4f}  v: {train_stats['value_loss']:.4f}) | "
            f"Val   loss: {val_stats['loss']:.4f}  acc: {val_stats['accuracy']:.3f}  "
            f"(p: {val_stats['policy_loss']:.4f}  v: {val_stats['value_loss']:.4f}) | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        checkpoint_state = {
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "train_stats":   train_stats,
            "val_stats":     val_stats,
        }

        # Salva sempre l'ultimo
        save_checkpoint(checkpoint_state, os.path.join(CHECKPOINT_DIR, "last.pt"))

        # Salva il best se migliora
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            checkpoint_state["best_val_loss"] = best_val_loss
            save_checkpoint(checkpoint_state, os.path.join(CHECKPOINT_DIR, "best.pt"))
            tqdm.write(f"  ★ Nuovo best val loss: {best_val_loss:.4f}")

        epoch_bar.set_postfix({
            "val_loss": f"{val_stats['loss']:.4f}",
            "val_acc":  f"{val_stats['accuracy']:.3f}",
            "best":     f"{best_val_loss:.4f}",
        })

    print(f"\nTraining completato. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
