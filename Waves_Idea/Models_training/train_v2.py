"""
train_v2.py
------------
Fair comparison experiment: baseline vs fields with precomputed HDF5.

Improvements over train.py:
  - Fields read from HDF5 (no BFS bottleneck — same speed as baseline)
  - Same exact positions for both variants
  - Gradient accumulation for larger effective batch size
  - More epochs with longer cosine annealing
  - Saves full training curves to JSON

Usage:
  python train_v2.py --csv your_dataset.csv --h5 fields.h5 --samples 500000 --epochs 25
"""

import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from MLChess import create_matched_dataloaders
from model import ChessNet, AlphaZeroLoss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def top_k_accuracy(logits, targets, k=1):
    valid = targets >= 0
    if not valid.any():
        return 0.0
    _, top_k = logits[valid].topk(k, dim=1)
    correct  = top_k.eq(targets[valid].unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


# ---------------------------------------------------------------------------
# One epoch with optional gradient accumulation
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device,
              train=True, accum_steps=1):
    model.train(train)
    context = torch.enable_grad() if train else torch.no_grad()

    total_loss = policy_loss_sum = value_loss_sum = 0.0
    top1_sum = top3_sum = top5_sum = 0.0
    n_batches = 0

    with context:
        if train:
            optimizer.zero_grad(set_to_none=True)

        for step, (boards, moves, masks, evals) in enumerate(loader):
            boards = boards.to(device, non_blocking=True)
            moves  = moves.to(device,  non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)
            evals  = evals.to(device,  non_blocking=True)

            policy, value = model(boards, masks)
            loss, p_loss, v_loss = criterion(policy, value, moves, evals)

            if train:
                (loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss      += loss.item()
            policy_loss_sum += p_loss.item()
            value_loss_sum  += v_loss.item()
            top1_sum        += top_k_accuracy(policy, moves, k=1)
            top3_sum        += top_k_accuracy(policy, moves, k=3)
            top5_sum        += top_k_accuracy(policy, moves, k=5)
            n_batches       += 1

    n = max(n_batches, 1)
    return {
        'loss':        total_loss      / n,
        'policy_loss': policy_loss_sum / n,
        'value_loss':  value_loss_sum  / n,
        'top1':        top1_sum        / n,
        'top3':        top3_sum        / n,
        'top5':        top5_sum        / n,
    }


# ---------------------------------------------------------------------------
# Train one variant
# ---------------------------------------------------------------------------

def train_variant(
    name:        str,
    in_channels: int,
    trainloader,
    valloader,
    epochs:      int,
    lr:          float,
    accum_steps: int,
    device:      torch.device,
    save_dir:    Path,
):
    print(f'\n{"="*55}')
    print(f'  Training: {name.upper()}  ({in_channels} input channels)')
    print(f'{"="*55}')

    model     = ChessNet(in_channels=in_channels).to(device)
    criterion = AlphaZeroLoss(value_lambda=1.0)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 20)

    print(f'Parameters:      {model.count_parameters():,}')
    print(f'Effective batch: {256 * accum_steps}  (accum_steps={accum_steps})')

    history       = []
    best_top1     = 0.0
    best_path     = save_dir / f'best_{name}.pt'

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_m = run_epoch(model, trainloader, criterion, optimizer,
                            device, train=True,  accum_steps=accum_steps)
        val_m   = run_epoch(model, valloader,   criterion, optimizer,
                            device, train=False, accum_steps=1)
        scheduler.step()

        elapsed = time.time() - t0

        print(
            f'Ep {epoch:>2}/{epochs}  [{elapsed:.0f}s]  '
            f'tr_loss={train_m["loss"]:.4f}  '
            f'val_loss={val_m["loss"]:.4f}  '
            f'top1={val_m["top1"]*100:.2f}%  '
            f'top3={val_m["top3"]*100:.2f}%  '
            f'top5={val_m["top5"]*100:.2f}%'
        )

        history.append({'epoch': epoch, 'train': train_m, 'val': val_m})

        if val_m['top1'] > best_top1:
            best_top1 = val_m['top1']
            torch.save({
                'epoch':      epoch,
                'variant':    name,
                'state_dict': model.state_dict(),
                'val_top1':   best_top1,
            }, best_path)

    print(f'\nBest val top-1: {best_top1*100:.2f}%')

    with open(save_dir / f'history_{name}.json', 'w') as f:
        json.dump(history, f, indent=2)

    return history, best_top1


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(hist_base, hist_fields, save_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [h['epoch'] for h in hist_base]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    configs = [
        ('top1',        'Top-1 Accuracy (%)', True),
        ('top3',        'Top-3 Accuracy (%)', True),
        ('top5',        'Top-5 Accuracy (%)', True),
        ('loss',        'Val Loss',           False),
    ]

    for ax, (metric, title, is_pct) in zip(axes.flat, configs):
        scale = 100 if is_pct else 1
        base_v   = [h['val'][metric] * scale for h in hist_base]
        fields_v = [h['val'][metric] * scale for h in hist_fields]

        ax.plot(epochs, base_v,   'o-', label='Baseline (13ch)',
                color='steelblue', linewidth=2, markersize=4)
        ax.plot(epochs, fields_v, 's-', label='Fields (16ch)',
                color='firebrick', linewidth=2, markersize=4)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)

        # Final value annotations
        for vals, color in [(base_v, 'steelblue'), (fields_v, 'firebrick')]:
            ax.annotate(f'{vals[-1]:.2f}',
                        (epochs[-1], vals[-1]),
                        xytext=(4, 0), textcoords='offset points',
                        fontsize=9, color=color)

    plt.suptitle('Baseline vs Influence Fields — Fair Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = save_dir / 'comparison_v2.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Plot saved: {out}')
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',     required=True,              help='CSV dataset path')
    parser.add_argument('--h5',      required=True,              help='Precomputed fields HDF5')
    parser.add_argument('--samples', type=int,   default=500_000)
    parser.add_argument('--epochs',  type=int,   default=25)
    parser.add_argument('--batch',   type=int,   default=256)
    parser.add_argument('--accum',   type=int,   default=2,      help='Gradient accum steps')
    parser.add_argument('--lr',      type=float, default=1e-3)
    parser.add_argument('--workers', type=int,   default=4)
    parser.add_argument('--outdir',  type=str,   default='runs_v2')
    args = parser.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.outdir)
    save_dir.mkdir(exist_ok=True)

    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU:   {torch.cuda.get_device_name(0)}')
        print(f'VRAM:  {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    print()

    # Create matched dataloaders — same positions for both variants
    train_base, val_base, train_fields, val_fields, _ = create_matched_dataloaders(
        csv_path   = args.csv,
        h5_path    = args.h5,
        n_samples  = args.samples,
        batch_size = args.batch,
        n_workers  = args.workers,
    )

    shared = dict(
        epochs      = args.epochs,
        lr          = args.lr,
        accum_steps = args.accum,
        device      = device,
        save_dir    = save_dir,
    )

    hist_base,   top1_base   = train_variant(
        'baseline', 13, train_base,   val_base,   **shared)
    hist_fields, top1_fields = train_variant(
        'fields',   16, train_fields, val_fields, **shared)

    # Summary
    delta = (top1_fields - top1_base) * 100
    print(f'\n{"="*55}')
    print(f'  RISULTATI FINALI')
    print(f'{"="*55}')
    print(f'  Baseline (13ch):    top-1 = {top1_base*100:.2f}%')
    print(f'  Fields   (16ch):    top-1 = {top1_fields*100:.2f}%')
    print(f'  Differenza:         {delta:+.2f} pp')
    print()
    if delta >= 1.0:
        print('  → I campi migliorano significativamente la policy.')
    elif delta >= 0.2:
        print('  → Miglioramento lieve ma consistente.')
    elif delta >= -0.2:
        print('  → Risultati equivalenti.')
    else:
        print('  → I campi non aiutano in questo setting.')

    summary = {
        'baseline_top1': top1_base,
        'fields_top1':   top1_fields,
        'delta_pp':      delta,
        'config':        vars(args),
    }
    with open(save_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    plot_comparison(hist_base, hist_fields, save_dir)


if __name__ == '__main__':
    main()
