"""
train_phase2.py
---------------
Fase 2 del training del modello chess AI.

Pipeline:
  Fase 2a — backbone congelato, allena solo ChessValuePolicy (freeze_epochs)
  Fase 2b — tutto sbloccato, lr differenziato (backbone lr basso, MLP lr normale)

Dipendenze esterne:
  - MLChess: ChessDataset, ChessTransform, collate_fn, generate_all_legal_move_vocab
  - model.py (o notebook): FullChessModel

Uso:
  python train_phase2.py
  python train_phase2.py --checkpoint chess_phase2_last.pt   # riprendi
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from MLChess import (
    ChessDataset,
    ChessTransform,
    collate_fn,
    generate_all_legal_move_vocab,
    FullChessModel
)


# ─────────────────────────────────────────────────────────────────────────────
# Iperparametri di default
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT = dict(
    csv_file        = "../over_mate_1_tactic_evals.csv",
    phase1_ckpt     = "chess_phase1_best.pt",
    batch_size      = 512,
    freeze_epochs   = 7,        # epoch con backbone congelato (fase 2a)
    total_epochs    = 37,       # epoch totali (freeze + unfreeze)
    lr_mlp          = 1e-4,     # lr per value/policy head
    lr_backbone     = 1e-5,     # lr per backbone nella fase 2b
    weight_decay    = 1e-4,
    lambda_policy   = 1.0,
    lambda_value    = 1.0,
    patience        = 5,        # ReduceLROnPlateau
    scheduler_factor= 0.5,
    grad_clip       = 1.0,
    num_workers     = 4,
    best_ckpt       = "chess_phase2_best.pt",
    last_ckpt       = "chess_phase2_last.pt",
)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def freeze_backbone(model: FullChessModel) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone(model: FullChessModel) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = True


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    move_vocab      = generate_all_legal_move_vocab()
    transform       = ChessTransform(move_vocab=move_vocab)

    train_ds = ChessDataset(cfg["csv_file"], move_vocab, split="train",      transform=transform)
    val_ds   = ChessDataset(cfg["csv_file"], move_vocab, split="validation", transform=transform)
    test_ds  = ChessDataset(cfg["csv_file"], move_vocab, split="test",       transform=transform)

    kwargs = dict(collate_fn=collate_fn, num_workers=cfg["num_workers"], pin_memory=True)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False, **kwargs)

    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader


def build_optimizer(model: FullChessModel, cfg: dict, frozen: bool) -> torch.optim.Optimizer:
    """
    frozen=True  → ottimizza solo la vp_head  (fase 2a)
    frozen=False → ottimizza tutto con lr differenziato (fase 2b)
    """
    if frozen:
        return torch.optim.Adam(
            model.vp_head.parameters(),
            lr=cfg["lr_mlp"],
            weight_decay=cfg["weight_decay"],
        )
    else:
        return torch.optim.Adam(
            [
                {"params": model.backbone.parameters(), "lr": cfg["lr_backbone"]},
                {"params": model.decoder.parameters(),  "lr": cfg["lr_backbone"]},
                {"params": model.vp_head.parameters(),  "lr": cfg["lr_mlp"]},
            ],
            weight_decay=cfg["weight_decay"],
        )


def compute_policy_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy sulla policy (mosse legali già mascherate in logits)."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def save_checkpoint(path: str, epoch: int, model: FullChessModel,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
                    metrics: dict, best_val_loss: float) -> None:
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics":              metrics,
            "best_val_loss":        best_val_loss,
        },
        path,
    )


def load_checkpoint(path: str, model: FullChessModel,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
                    device: torch.device) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch    = ckpt["epoch"]
    best_val_loss  = ckpt.get("best_val_loss", float("inf"))
    print(f"Ripreso da epoch {start_epoch}  |  best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss


# ─────────────────────────────────────────────────────────────────────────────
# Loop singola epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:       FullChessModel,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer | None,
    cfg:         dict,
    device:      torch.device,
    epoch:       int,
    total_epochs:int,
    tag:         str,
) -> dict:
    """
    Esegue una epoch di training (optimizer != None) o validazione.
    Ritorna un dict con le metriche medie.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    tot_loss = tot_policy = tot_value = tot_acc = 0.0
    n_batches = 0

    bar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [{tag}]", leave=False)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for boards, moves, mask, evals in bar:
            boards = boards.to(device)
            moves  = moves.to(device)
            mask   = mask.to(device)
            evals  = evals.to(device)

            policy_logits, value = model.forward_phase2(boards, mask)

            loss_policy = F.cross_entropy(policy_logits, moves)
            loss_value  = F.mse_loss(value.squeeze(-1), evals)
            loss        = cfg["lambda_policy"] * loss_policy + cfg["lambda_value"] * loss_value

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()

            acc = compute_policy_accuracy(policy_logits, moves)

            tot_loss   += loss.item()
            tot_policy += loss_policy.item()
            tot_value  += loss_value.item()
            tot_acc    += acc
            n_batches  += 1

            bar.set_postfix(
                loss   = f"{loss.item():.4f}",
                policy = f"{loss_policy.item():.4f}",
                value  = f"{loss_value.item():.4f}",
                acc    = f"{acc*100:.1f}%",
            )

    return dict(
        loss   = tot_loss   / n_batches,
        policy = tot_policy / n_batches,
        value  = tot_value  / n_batches,
        acc    = tot_acc    / n_batches,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training principale
# ─────────────────────────────────────────────────────────────────────────────

def train_phase2(cfg: dict, resume_checkpoint: str | None = None) -> FullChessModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── modello ───────────────────────────────────────────────────────────────
    model = FullChessModel().to(device)

    # Carica pesi dalla fase 1
    phase1_ckpt = torch.load(cfg["phase1_ckpt"], map_location=device)
    model.load_state_dict(phase1_ckpt["model_state_dict"])
    print(f"Pesi fase 1 caricati da '{cfg['phase1_ckpt']}'")

    # ── dati ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # ── stato iniziale: backbone congelato ───────────────────────────────────
    freeze_backbone(model)
    frozen    = True
    optimizer = build_optimizer(model, cfg, frozen=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg["patience"], factor=cfg["scheduler_factor"]
    )

    start_epoch   = 0
    best_val_loss = float("inf")

    # ── riprendi da checkpoint se richiesto ──────────────────────────────────
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        start_epoch, best_val_loss = load_checkpoint(
            resume_checkpoint, model, optimizer, scheduler, device
        )
        # Ripristina stato freeze in base all'epoch
        if start_epoch >= cfg["freeze_epochs"]:
            unfreeze_backbone(model)
            frozen    = False
            optimizer = build_optimizer(model, cfg, frozen=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=cfg["patience"], factor=cfg["scheduler_factor"]
            )
            # Ricarica stati optimizer/scheduler dal checkpoint per fase 2b
            ckpt = torch.load(resume_checkpoint, map_location=device)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # ── training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, cfg["total_epochs"] + 1):

        # Transizione 2a → 2b
        if frozen and epoch > cfg["freeze_epochs"]:
            print(f"\n{'─'*60}")
            print(f"  Epoch {epoch}: sblocco backbone → fase 2b (lr differenziato)")
            print(f"{'─'*60}")
            unfreeze_backbone(model)
            frozen    = False
            optimizer = build_optimizer(model, cfg, frozen=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=cfg["patience"], factor=cfg["scheduler_factor"]
            )

        phase_tag = "2a-frozen" if frozen else "2b-unfrz"

        # Train
        train_metrics = run_epoch(
            model, train_loader, optimizer, cfg, device,
            epoch, cfg["total_epochs"], f"Train {phase_tag}"
        )

        # Validation
        val_metrics = run_epoch(
            model, val_loader, None, cfg, device,
            epoch, cfg["total_epochs"], f"Val   {phase_tag}"
        )

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[-1]["lr"]  # lr della vp_head

        # Stampa riassunto
        print(
            f"Epoch {epoch:3d}/{cfg['total_epochs']} [{phase_tag}] | "
            f"LR: {current_lr:.2e} | "
            f"Train loss: {train_metrics['loss']:.4f} "
            f"(pol: {train_metrics['policy']:.4f}, val: {train_metrics['value']:.4f}, "
            f"acc: {train_metrics['acc']*100:.1f}%) | "
            f"Val loss: {val_metrics['loss']:.4f} "
            f"(pol: {val_metrics['policy']:.4f}, val: {val_metrics['value']:.4f}, "
            f"acc: {val_metrics['acc']*100:.1f}%)"
        )

        metrics = {"train": train_metrics, "val": val_metrics}

        # Salva ultimo
        save_checkpoint(cfg["last_ckpt"], epoch, model, optimizer, scheduler, metrics, best_val_loss)

        # Salva migliore
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(cfg["best_ckpt"], epoch, model, optimizer, scheduler, metrics, best_val_loss)
            print(f"  → Nuovo best model salvato (val_loss={best_val_loss:.4f})")

    # ── test finale ──────────────────────────────────────────────────────────
    print("\nValutazione sul test set con il best model...")
    best_ckpt = torch.load(cfg["best_ckpt"], map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = run_epoch(
        model, test_loader, None, cfg, device,
        cfg["total_epochs"], cfg["total_epochs"], "Test"
    )
    print(
        f"Test | loss: {test_metrics['loss']:.4f} | "
        f"policy: {test_metrics['policy']:.4f} | "
        f"value: {test_metrics['value']:.4f} | "
        f"acc: {test_metrics['acc']*100:.1f}%"
    )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess AI — Phase 2 Training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path a un checkpoint di fase 2 da cui riprendere")
    parser.add_argument("--csv",        type=str, default=DEFAULT["csv_file"])
    parser.add_argument("--phase1",     type=str, default=DEFAULT["phase1_ckpt"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT["batch_size"])
    parser.add_argument("--freeze-epochs", type=int, default=DEFAULT["freeze_epochs"])
    parser.add_argument("--total-epochs", type=int, default=DEFAULT["total_epochs"])
    args = parser.parse_args()

    cfg = DEFAULT.copy()
    cfg["csv_file"]      = args.csv
    cfg["phase1_ckpt"]   = args.phase1
    cfg["batch_size"]    = args.batch_size
    cfg["freeze_epochs"] = args.freeze_epochs
    cfg["total_epochs"]  = args.total_epochs

    train_phase2(cfg, resume_checkpoint=args.checkpoint)
