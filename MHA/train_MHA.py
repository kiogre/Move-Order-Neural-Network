import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from tqdm import tqdm

from MLChess import create_dataloaders_tensor, ChessMHA, ChessMHA_2, ChessMHA_3

# ── Configurazione ────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_FILE    = "../over_mate_1_tactic_evals.csv"
BATCH_SIZE  = 1024
EPOCHS      = 30
LR          = 1e-3
CHECKPOINT  = "chess_mha_3_checkpoint.pt"

# Pesi delle due loss
LAMBDA_VALUE  = 1.0
LAMBDA_POLICY = 1.0


# ── Funzioni di supporto ──────────────────────────────────────────────────────

def policy_loss_fn(logits: torch.Tensor,
                   targets: torch.Tensor) -> torch.Tensor:
    """
    La maschera è già applicata dentro il forward del modello
    con masked_fill(-inf), quindi qui basta la cross-entropy standard.
    """
    return nn.CrossEntropyLoss()(logits, targets)


def run_epoch(model, loader, optimizer, train: bool):
    model.train() if train else model.eval()

    total_loss = total_value_loss = total_policy_loss = 0.0
    correct = 0
    total   = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for boards, moves, mask, results in tqdm(loader):
            boards  = boards.to(DEVICE)
            moves   = moves.to(DEVICE)
            mask    = mask.to(DEVICE)
            results = results.to(DEVICE)

            value_pred, policy_pred = model(boards, mask)

            # Value loss — regressione sulla valutazione normalizzata in [-1, 1]
            v_loss = nn.MSELoss()(value_pred.squeeze(-1), results)

            # Policy loss — la maschera è già applicata dentro il modello
            p_loss = policy_loss_fn(policy_pred, moves)

            loss = LAMBDA_VALUE * v_loss + LAMBDA_POLICY * p_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Statistiche
            batch_size     = boards.size(0)
            total_loss    += loss.item()        * batch_size
            total_value_loss  += v_loss.item()  * batch_size
            total_policy_loss += p_loss.item()  * batch_size

            # Accuracy policy (top-1) — policy_pred ha già -inf sulle mosse illegali
            predicted_move = policy_pred.argmax(dim=-1)
            correct += (predicted_move == moves).sum().item()
            total   += batch_size

    n = total
    return (total_loss / n,
            total_value_loss / n,
            total_policy_loss / n,
            correct / n)


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "val_loss":   val_loss,
    }, path)
    print(f"  → checkpoint salvato (epoch {epoch}, val_loss {val_loss:.4f})")


def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  → checkpoint caricato (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")
    return ckpt["epoch"], ckpt["val_loss"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")

    # Dataset
    trainloader, valloader, testloader, move_vocab = create_dataloaders_tensor(
        name_file=CSV_FILE,
        batch_size=BATCH_SIZE,
        num_workers = 4,
        pin_memory=True
    )

    # Modello
    model = ChessMHA_3().to(DEVICE)
    print(f"Parametri totali: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    start_epoch = 0
    best_val_loss = float("inf")

    # Riprendi da checkpoint se esiste
    if os.path.exists(CHECKPOINT):
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, CHECKPOINT)

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        tr_loss, tr_v, tr_p, tr_acc = run_epoch(model, trainloader, optimizer, train=True)
        vl_loss, vl_v, vl_p, vl_acc = run_epoch(model, valloader,   optimizer, train=False)

        scheduler.step(vl_loss)

        print(
            f"Epoch {epoch+1:03d}/{EPOCHS} | "
            f"Train  loss {tr_loss:.4f}  value {tr_v:.4f}  policy {tr_p:.4f}  acc {tr_acc:.3f} | "
            f"Val    loss {vl_loss:.4f}  value {vl_v:.4f}  policy {vl_p:.4f}  acc {vl_acc:.3f}"
        )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            save_checkpoint(model, optimizer, epoch + 1, best_val_loss, CHECKPOINT)

    # Test finale
    print("\nValutazione sul test set...")
    te_loss, te_v, te_p, te_acc = run_epoch(model, testloader, optimizer, train=False)
    print(
        f"Test | loss {te_loss:.4f}  value {te_v:.4f}  policy {te_p:.4f}  acc {te_acc:.3f}"
    )


if __name__ == "__main__":
    main()
