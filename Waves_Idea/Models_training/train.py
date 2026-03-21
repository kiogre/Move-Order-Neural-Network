"""
train.py
---------
Training script for the dual-head ChessNet.
Runs two experiments back-to-back:
  1. Baseline  — input (13, 8, 8)
  2. With fields — input (16, 8, 8)

Usage:
  python train.py --csv your_dataset.csv --samples 500000

Full options:
  python train.py --csv your_dataset.csv --samples 500000 --epochs 10
                  --batch 256 --lr 1e-3 --alpha 0.9 --workers 4
"""

import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from MLChess import ChessDataset, ChessTransform, generate_all_legal_move_vocab, collate_fn, ChessDatasetWithFields, ChessTransformWithFields
from model import ChessNet, AlphaZeroLoss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def top_k_accuracy(logits, targets, k=1):
    """Accuracy among positions where target >= 0."""
    valid = targets >= 0
    if not valid.any():
        return 0.0
    logits_v  = logits[valid]
    targets_v = targets[valid]
    _, top_k  = logits_v.topk(k, dim=1)
    correct   = top_k.eq(targets_v.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    context = torch.enable_grad() if train else torch.no_grad()

    total_loss = policy_loss_sum = value_loss_sum = 0.0
    top1_sum = top3_sum = top5_sum = 0.0
    n_batches = 0

    with context:
        for boards, moves, masks, evals in loader:
            boards = boards.to(device, non_blocking=True)
            moves  = moves.to(device,  non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)
            evals  = evals.to(device,  non_blocking=True)

            policy, value = model(boards, masks)
            loss, p_loss, v_loss = criterion(policy, value, moves, evals)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss      += loss.item()
            policy_loss_sum += p_loss.item()
            value_loss_sum  += v_loss.item()
            top1_sum        += top_k_accuracy(policy, moves, k=1)
            top3_sum        += top_k_accuracy(policy, moves, k=3)
            top5_sum        += top_k_accuracy(policy, moves, k=5)
            n_batches       += 1

    n = max(n_batches, 1)
    return {
        'loss':         total_loss      / n,
        'policy_loss':  policy_loss_sum / n,
        'value_loss':   value_loss_sum  / n,
        'top1':         top1_sum        / n,
        'top3':         top3_sum        / n,
        'top5':         top5_sum        / n,
    }


# ---------------------------------------------------------------------------
# Full training run for one model variant
# ---------------------------------------------------------------------------

def train_model(
    variant:    str,          # 'baseline' or 'fields'
    csv_path:   str,
    n_samples:  int,
    epochs:     int,
    batch_size: int,
    lr:         float,
    alpha:      float,
    n_workers:  int,
    device:     torch.device,
    save_dir:   Path,
):
    print(f'\n{"="*55}')
    print(f'  Training: {variant.upper()}')
    print(f'{"="*55}')

    move_vocab = generate_all_legal_move_vocab()

    # Dataset
    if variant == 'baseline':
        transform   = ChessTransform(move_vocab=move_vocab)
        DatasetCls  = ChessDataset
        in_channels = 13
    else:
        transform   = ChessTransformWithFields(move_vocab=move_vocab, alpha=alpha)
        DatasetCls  = ChessDatasetWithFields
        in_channels = 16

    # Subset: take first n_samples rows (already shuffled by CSV order)
    # We'll do the split manually to keep the same positions across variants
    import pandas as pd
    df_sub = pd.read_csv(csv_path).sample(
        n=min(n_samples, sum(1 for _ in open(csv_path)) - 1),
        random_state=42
    )
    sub_path = save_dir / f'_subset_{variant}.csv'
    df_sub.to_csv(sub_path, index=False)

    train_end = int(len(df_sub) * 0.85)  # 85/15 train/val split for experiment
    df_train = df_sub.iloc[:train_end]
    df_val   = df_sub.iloc[train_end:]

    train_path = save_dir / f'_train_{variant}.csv'
    val_path   = save_dir / f'_val_{variant}.csv'
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)

    if variant == 'baseline':
        trainset = ChessDataset(str(train_path), move_vocab, split='train',
                                transform=transform)
        valset   = ChessDataset(str(val_path),   move_vocab, split='train',
                                transform=transform)
    else:
        trainset = ChessDatasetWithFields(str(train_path), move_vocab, split='train',
                                          transform=transform)
        valset   = ChessDatasetWithFields(str(val_path),   move_vocab, split='train',
                                          transform=transform)

    g = torch.Generator()
    g.manual_seed(42)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=n_workers,
        pin_memory=True, generator=g,
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=n_workers,
        pin_memory=True,
    )

    # Model
    model     = ChessNet(in_channels=in_channels).to(device)
    criterion = AlphaZeroLoss(value_lambda=1.0)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/20)

    print(f'Parameters: {model.count_parameters():,}')
    print(f'Train: {len(trainset):,}  |  Val: {len(valset):,}')
    print(f'Batches/epoch: {len(trainloader):,}')
    print()

    history = []
    best_val_top1 = 0.0
    best_path     = save_dir / f'best_{variant}.pt'

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(model, trainloader, criterion, optimizer, device, train=True)
        val_metrics   = run_epoch(model, valloader,   criterion, optimizer, device, train=False)
        scheduler.step()

        elapsed = time.time() - t0

        print(
            f'Epoch {epoch:>2}/{epochs}  '
            f'[{elapsed:.0f}s]  '
            f'train_loss={train_metrics["loss"]:.4f}  '
            f'val_loss={val_metrics["loss"]:.4f}  '
            f'val_top1={val_metrics["top1"]*100:.2f}%  '
            f'val_top3={val_metrics["top3"]*100:.2f}%  '
            f'val_top5={val_metrics["top5"]*100:.2f}%'
        )

        epoch_log = {
            'epoch': epoch,
            'train': train_metrics,
            'val':   val_metrics,
        }
        history.append(epoch_log)

        if val_metrics['top1'] > best_val_top1:
            best_val_top1 = val_metrics['top1']
            torch.save({
                'epoch':      epoch,
                'variant':    variant,
                'state_dict': model.state_dict(),
                'val_top1':   best_val_top1,
            }, best_path)

    # Save history
    hist_path = save_dir / f'history_{variant}.json'
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nBest val top-1: {best_val_top1*100:.2f}%')
    print(f'Model saved to: {best_path}')

    # Cleanup temp CSVs
    for p in [sub_path, train_path, val_path]:
        p.unlink(missing_ok=True)

    return history, best_val_top1


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def plot_comparison(hist_baseline, hist_fields, save_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping plot.')
        return

    epochs    = [h['epoch'] for h in hist_baseline]
    metrics   = ['top1', 'top3', 'top5', 'loss']
    titles    = ['Top-1 Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy', 'Val Loss']

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, metric, title in zip(axes.flat, metrics, titles):
        base_vals   = [h['val'][metric] for h in hist_baseline]
        fields_vals = [h['val'][metric] for h in hist_fields]

        scale = 100 if metric != 'loss' else 1
        ax.plot(epochs, [v * scale for v in base_vals],
                'o-', label='Baseline (13 ch)', color='steelblue')
        ax.plot(epochs, [v * scale for v in fields_vals],
                's-', label='With fields (16 ch)', color='firebrick')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('%' if metric != 'loss' else 'Loss')
        ax.legend()
        ax.grid(alpha=0.3)

        # Annotate final values
        ax.annotate(f'{base_vals[-1]*scale:.2f}',
                    (epochs[-1], base_vals[-1]*scale),
                    textcoords='offset points', xytext=(5, 0), fontsize=9)
        ax.annotate(f'{fields_vals[-1]*scale:.2f}',
                    (epochs[-1], fields_vals[-1]*scale),
                    textcoords='offset points', xytext=(5, 0), fontsize=9,
                    color='firebrick')

    plt.suptitle('Baseline vs With Influence Fields', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = save_dir / 'comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f'Plot saved to: {plot_path}')
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',     required=True)
    parser.add_argument('--samples', type=int,   default=500_000)
    parser.add_argument('--epochs',  type=int,   default=10)
    parser.add_argument('--batch',   type=int,   default=256)
    parser.add_argument('--lr',      type=float, default=1e-3)
    parser.add_argument('--alpha',   type=float, default=0.9)
    parser.add_argument('--workers', type=int,   default=4)
    parser.add_argument('--outdir',  type=str,   default='runs')
    args = parser.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.outdir)
    save_dir.mkdir(exist_ok=True)

    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    kwargs = dict(
        csv_path   = args.csv,
        n_samples  = args.samples,
        epochs     = args.epochs,
        batch_size = args.batch,
        lr         = args.lr,
        alpha      = args.alpha,
        n_workers  = args.workers,
        device     = device,
        save_dir   = save_dir,
    )

    # Train baseline
    hist_baseline, top1_baseline = train_model('baseline', **kwargs)

    # Train with fields
    hist_fields, top1_fields = train_model('fields', **kwargs)

    # Summary
    print(f'\n{"="*55}')
    print(f'  RISULTATI FINALI')
    print(f'{"="*55}')
    print(f'  Baseline (13ch):     top-1 = {top1_baseline*100:.2f}%')
    print(f'  With fields (16ch):  top-1 = {top1_fields*100:.2f}%')
    delta = (top1_fields - top1_baseline) * 100
    sign  = '+' if delta >= 0 else ''
    print(f'  Differenza:          {sign}{delta:.2f}pp')
    print()
    if delta > 0:
        print('  → I campi di influenza MIGLIORANO la policy network.')
    elif delta > -0.5:
        print('  → Risultati equivalenti. I campi non danneggiano.')
    else:
        print('  → I campi non aiutano in questo setting.')

    # Save summary
    summary = {
        'baseline_top1': top1_baseline,
        'fields_top1':   top1_fields,
        'delta_pp':      delta,
        'config':        vars(args),
    }
    with open(save_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    plot_comparison(hist_baseline, hist_fields, save_dir)


if __name__ == '__main__':
    main()
