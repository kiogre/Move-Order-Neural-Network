"""
train_vqvae.py
==============
Training loop completo per ChessVQVAE.

Caratteristiche:
  - Usa direttamente ChessDataset / ChessTransform da data_organization_tensor.py
  - Mixed precision (torch.amp) per sfruttare la RTX 3060
  - Checkpointing automatico (miglior val loss + periodico ogni N epoch)
  - Logging su file + console con metriche complete
  - Codebook monitoring: uso, perplexity, dead code reset automatico
  - Early stopping configurabile

Uso:
    python train_vqvae.py --csv path/to/file.csv --epochs 50 --batch_size 256
"""

import os
import math
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from MLChess import create_dataloaders_tensor
from vqvae_chess import ChessVQVAE


# ─────────────────────────────────────────────
# Configurazione logging
# ─────────────────────────────────────────────

def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("vqvae_chess")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    # Console — usa tqdm.write per non spezzare le barre di avanzamento
    class TqdmHandler(logging.StreamHandler):
        def emit(self, record):
            tqdm.write(self.format(record))

    ch = TqdmHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(log_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
# Metriche codebook
# ─────────────────────────────────────────────

def codebook_perplexity(indices: torch.Tensor, num_embeddings: int) -> float:
    """
    Perplexity del codebook: misura quanti vettori vengono
    effettivamente usati in modo uniforme.
    Valore ideale ≈ num_embeddings (tutti usati equamente).
    Valore basso → collasso del codebook.
    """
    one_hot = torch.zeros(num_embeddings, device=indices.device)
    one_hot.scatter_add_(0, indices.flatten(),
                         torch.ones_like(indices.flatten(), dtype=torch.float))
    probs = one_hot / one_hot.sum()
    # Entropia
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum()
    return math.exp(entropy.item())


# ─────────────────────────────────────────────
# Accuracy ricostruzione pezzi
# ─────────────────────────────────────────────

def piece_accuracy(x_recon: torch.Tensor, x_true: torch.Tensor) -> float:
    """
    Accuracy casella per casella usando argmax categorico.
    x_recon[:, :13] = logit categorici (0=vuoto, 1-12=pezzi)
    x_true[:, :12]  = piani binari target
    """
    pred_idx   = x_recon[:, :13].argmax(dim=1)                    # (B, 8, 8)
    piece_idx  = x_true[:, :12].argmax(dim=1) + 1                 # (B, 8, 8)
    is_empty   = x_true[:, :12].sum(dim=1) == 0                   # (B, 8, 8)
    target_idx = torch.where(is_empty,
                             torch.zeros_like(piece_idx),
                             piece_idx)
    return (pred_idx == target_idx).float().mean().item()


def turn_accuracy(x_recon: torch.Tensor, x_true: torch.Tensor) -> float:
    # Piano 13 del decoder = turno
    pred  = (torch.sigmoid(x_recon[:, 13:14, 0, 0]) > 0.5).float()
    truth = x_true[:, 12:13, 0, 0]
    return (pred == truth).float().mean().item()


# ─────────────────────────────────────────────
# Un singolo step di training
# ─────────────────────────────────────────────

def train_step(model, batch, optimizer, scaler, device,
               grad_accum: int = 1, accum_step: int = 0):
    """
    grad_accum: quanti step accumulare prima di fare optimizer.step()
    accum_step: step corrente dentro l'accumulazione (0-based)
    """
    positions, _, _, result = batch
    positions = positions.to(device, non_blocking=True)
    result    = result.to(device,    non_blocking=True)

    # zero_grad solo al primo step dell'accumulazione
    if accum_step == 0:
        optimizer.zero_grad(set_to_none=True)

    with autocast("cuda"):
        out  = model(positions, aux_target=result)
        # Scala la loss per l'accumulazione (media, non somma)
        loss = out["loss"] / grad_accum

    scaler.scale(loss).backward()

    # Optimizer step solo all'ultimo step dell'accumulazione
    if accum_step == grad_accum - 1:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    # Restituisce le metriche non scalate
    out["loss"] = out["loss"].detach()
    return out


# ─────────────────────────────────────────────
# Epoch di training
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scaler, device,
                logger, epoch, cfg, dead_reset_every=200):
    model.train()
    total_loss = total_recon = total_vq = total_commit = total_aux = 0.0
    total_piece_acc = total_turn_acc = 0.0
    n_batches = 0

    grad_accum = cfg.grad_accum
    pbar = tqdm(loader, desc=f"Epoch {epoch+1:3d}", dynamic_ncols=True, leave=True)
    for step, batch in enumerate(pbar):
        accum_step = step % grad_accum
        out = train_step(model, batch, optimizer, scaler, device,
                         grad_accum=grad_accum, accum_step=accum_step)

        positions = batch[0].to(device, non_blocking=True)
        total_loss    += out["loss"].item()
        total_recon   += out["recon_loss"].item()
        total_vq      += out["vq_loss"].item()
        total_commit  += out["commitment"].item()
        total_aux     += out["aux_loss"].item() if isinstance(out["aux_loss"], torch.Tensor) else 0.0
        total_piece_acc += piece_accuracy(out["x_recon"].detach(), positions)
        total_turn_acc  += turn_accuracy(out["x_recon"].detach(), positions)
        n_batches += 1

        # Aggiorna barra con medie correnti ogni 20 step
        if step % 20 == 0:
            n = max(n_batches, 1)
            pbar.set_postfix(
                loss=f"{total_loss/n:.4f}",
                recon=f"{total_recon/n:.4f}",
                vq=f"{total_vq/n:.4f}",
                refresh=False,
            )

        # Dead code reset periodico — salta sull'ultimo batch (è più piccolo
        # e farebbe reset spurio perché pochi campioni → molti vettori 'morti')
        is_last_step = (step == len(loader) - 1)
        if step % dead_reset_every == 0 and step > 0 and not is_last_step:
            with torch.no_grad():
                z_e_flat = (out["z_e"].detach()
                            .permute(0, 2, 3, 1)
                            .reshape(-1, out["z_e"].size(1)))
                n_reset = model.vq.reset_dead_codes(z_e_flat)
            if n_reset > 50:
                tqdm.write(f"  [step {step}] dead reset: {n_reset} vettori")

    n = max(n_batches, 1)
    return {
        "loss":      total_loss    / n,
        "recon":     total_recon   / n,
        "vq":        total_vq      / n,
        "commit":    total_commit  / n,
        "aux":       total_aux     / n,
        "piece_acc": total_piece_acc / n,
        "turn_acc":  total_turn_acc  / n,
    }


# ─────────────────────────────────────────────
# Epoch di validazione
# ─────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, device, num_embeddings):
    model.eval()
    total_loss = total_recon = total_vq = total_commit = total_aux = 0.0
    total_piece_acc = total_turn_acc = 0.0
    all_indices = []
    n_batches = 0

    for batch in tqdm(loader, desc="       Val", dynamic_ncols=True, leave=False):
        positions, _, _, result = batch
        positions = positions.to(device, non_blocking=True)

        result = result.to(device, non_blocking=True)
        with autocast("cuda"):
            out = model(positions, aux_target=result)

        total_loss    += out["loss"].item()
        total_recon   += out["recon_loss"].item()
        total_vq      += out["vq_loss"].item()
        total_commit  += out["commitment"].item()
        total_aux     += out["aux_loss"].item() if isinstance(out["aux_loss"], torch.Tensor) else 0.0
        total_piece_acc += piece_accuracy(out["x_recon"], positions)
        total_turn_acc  += turn_accuracy(out["x_recon"], positions)
        all_indices.append(out["indices"].cpu())
        n_batches += 1

    all_indices = torch.cat(all_indices)
    perplexity  = codebook_perplexity(all_indices, num_embeddings)
    usage_pct   = (all_indices.unique().numel() / num_embeddings) * 100

    n = max(n_batches, 1)
    return {
        "loss":       total_loss    / n,
        "recon":      total_recon   / n,
        "vq":         total_vq      / n,
        "commit":     total_commit  / n,
        "aux":        total_aux     / n,
        "piece_acc":  total_piece_acc / n,
        "turn_acc":   total_turn_acc  / n,
        "perplexity": perplexity,
        "usage_pct":  usage_pct,
    }


# ─────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────

def save_checkpoint(state: dict, path: Path):
    torch.save(state, path)


def load_checkpoint(path: Path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt["best_val_loss"]


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def train(cfg):
    # Directories
    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    log_dir  = Path(cfg.output_dir) / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir)
    logger.info("=" * 60)
    logger.info("ChessVQVAE - Training")
    logger.info("=" * 60)
    logger.info(f"Config: {vars(cfg)}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Dataloaders
    logger.info("Caricamento dataset...")
    trainloader, valloader, testloader, move_vocab = create_dataloaders_tensor(
        name_file=cfg.csv,
        batch_size=cfg.batch_size,
    )
    logger.info(f"Batch per epoca (train): {len(trainloader)}")

    # Modello
    model = ChessVQVAE(
        latent_dim=cfg.latent_dim,
        num_embeddings=cfg.num_embeddings,
        base_ch=cfg.base_ch,
        beta=cfg.beta,
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
        aux_weight=cfg.aux_weight,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parametri modello: {total_params:,}")

    # Ottimizzatore
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # Scheduler: cosine decay con warmup
    total_steps = cfg.epochs * len(trainloader)
    warmup_steps = cfg.warmup_epochs * len(trainloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(cfg.min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler("cuda")

    # Resume da checkpoint se richiesto
    start_epoch = 0
    best_val_loss = float("inf")
    early_stop_counter = 0

    if cfg.resume and Path(cfg.resume).exists():
        start_epoch, best_val_loss = load_checkpoint(
            Path(cfg.resume), model, optimizer, scaler)
        logger.info(f"Ripreso da {cfg.resume} (epoch {start_epoch})")

    # ── Training ──
    for epoch in range(start_epoch, cfg.epochs):
        logger.info(f"\n{'─'*60}")
        logger.info(f"Epoch {epoch+1}/{cfg.epochs}  |  LR: {scheduler.get_last_lr()[0]:.2e}")
        logger.info(f"{'─'*60}")

        t_epoch = time.time()

        train_metrics = train_epoch(
            model, trainloader, optimizer, scaler, device,
            logger, epoch, cfg,
            dead_reset_every=cfg.dead_reset_every,
        )
        # Advance scheduler una volta per step (già avanzato dentro train_step indirettamente)
        # Avanziamo qui a fine epoca per semplicità con LambdaLR
        for _ in range(len(trainloader)):
            scheduler.step()

        val_metrics = val_epoch(model, valloader, device, cfg.num_embeddings)

        epoch_time = time.time() - t_epoch

        # Log riepilogo epoch
        logger.info(
            f"\nEpoch {epoch+1} completata in {epoch_time:.0f}s\n"
            f"  TRAIN | loss {train_metrics['loss']:.4f} | "
            f"recon {train_metrics['recon']:.4f} | "
            f"vq {train_metrics['vq']:.4f} | "
            f"aux {train_metrics['aux']:.4f} | "
            f"piece_acc {train_metrics['piece_acc']:.4f} | "
            f"turn_acc {train_metrics['turn_acc']:.4f}\n"
            f"  VAL   | loss {val_metrics['loss']:.4f} | "
            f"recon {val_metrics['recon']:.4f} | "
            f"vq {val_metrics['vq']:.4f} | "
            f"aux {val_metrics['aux']:.4f} | "
            f"piece_acc {val_metrics['piece_acc']:.4f} | "
            f"turn_acc {val_metrics['turn_acc']:.4f}\n"
            f"  CODEBOOK | perplexity {val_metrics['perplexity']:.1f} / {cfg.num_embeddings} | "
            f"usage {val_metrics['usage_pct']:.1f}%"
        )

        # Attenzione: perplexity bassa → codebook collapse!
        if val_metrics["perplexity"] < cfg.num_embeddings * 0.05:
            logger.warning(
                f"  ⚠ Codebook collapse sospettato! "
                f"perplexity {val_metrics['perplexity']:.1f} < "
                f"{cfg.num_embeddings * 0.05:.0f}")

        # Checkpoint miglior val loss
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            early_stop_counter = 0
            save_checkpoint(
                {
                    "epoch":         epoch + 1,
                    "model":         model.state_dict(),
                    "optimizer":     optimizer.state_dict(),
                    "scaler":        scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "cfg":           vars(cfg),
                },
                ckpt_dir / "best.pt",
            )
            logger.info(f"  ✓ Miglior checkpoint salvato (val_loss={best_val_loss:.4f})")
        else:
            early_stop_counter += 1

        # Checkpoint periodico
        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(
                {
                    "epoch":         epoch + 1,
                    "model":         model.state_dict(),
                    "optimizer":     optimizer.state_dict(),
                    "scaler":        scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "cfg":           vars(cfg),
                },
                ckpt_dir / f"epoch_{epoch+1:04d}.pt",
            )

        # Early stopping
        if cfg.patience > 0 and early_stop_counter >= cfg.patience:
            logger.info(f"\nEarly stopping: {early_stop_counter} epoch senza miglioramenti.")
            break

    # ── Test finale ──
    logger.info("\n" + "=" * 60)
    logger.info("Test finale sul best checkpoint...")
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
    test_metrics = val_epoch(model, testloader, device, cfg.num_embeddings)
    logger.info(
        f"TEST | loss {test_metrics['loss']:.4f} | "
        f"piece_acc {test_metrics['piece_acc']:.4f} | "
        f"turn_acc {test_metrics['turn_acc']:.4f} | "
        f"perplexity {test_metrics['perplexity']:.1f} | "
        f"usage {test_metrics['usage_pct']:.1f}%"
    )

    logger.info("\nTraining completato.")
    return model


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train ChessVQVAE")

    # Dati
    p.add_argument("--csv",        type=str, required=True,
                   help="Path al file CSV con FEN,Move,Evaluation")
    p.add_argument("--batch_size", type=int, default=128,
                   help="Batch size (default 128 per evitare OOM su RTX 3060)")

    # Modello
    p.add_argument("--latent_dim",      type=int,   default=256)
    p.add_argument("--num_embeddings",  type=int,   default=512,
                   help="Dimensione codebook VQ")
    p.add_argument("--base_ch",         type=int,   default=128,
                   help="Canali base encoder/decoder")
    p.add_argument("--beta",            type=float, default=0.5,
                   help="Peso commitment loss (più alto = meno codebook collapse)")
    p.add_argument("--focal_alpha",     type=float, default=0.85,
                   help="Peso classe positiva nella focal loss")
    p.add_argument("--focal_gamma",     type=float, default=2.0,
                   help="Focusing parameter focal loss")

    # Training
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--min_lr_ratio",    type=float, default=0.1,
                   help="LR finale = lr * min_lr_ratio (cosine decay)")
    p.add_argument("--warmup_epochs",   type=int,   default=2,
                   help="Epoch di warmup lineare")
    p.add_argument("--dead_reset_every",type=int,   default=200,
                   help="Ogni quanti step fare dead code reset")
    p.add_argument("--grad_accum",      type=int,   default=4,
                   help="Gradient accumulation steps (batch effettivo = batch_size * grad_accum)")
    p.add_argument("--aux_weight",      type=float, default=0.5,
                   help="Peso della loss ausiliaria di valutazione posizionale")

    # Checkpointing / early stopping
    p.add_argument("--output_dir",  type=str, default="./runs/vqvae")
    p.add_argument("--save_every",  type=int, default=5,
                   help="Salva checkpoint ogni N epoch")
    p.add_argument("--patience",    type=int, default=10,
                   help="Early stopping: epoch senza miglioramento (0 = disattivato)")
    p.add_argument("--resume",      type=str, default="",
                   help="Path a un checkpoint da cui riprendere")

    return p.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
