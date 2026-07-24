"""
train_pointer.py — Training supervisionato JellyFishPointer.

Cambiamenti rispetto alla versione precedente:
  - Policy loss: cross-entropy manuale su probs (no NaN da -inf nel padding)
  - encode_board_fast invece di encode_board (28x piu veloce)
  - AMP (mixed precision): ~40% speedup
  - AdamW + weight_decay invece di Adam
  - CosineAnnealingLR invece di ReduceLROnPlateau
  - Split 98/2 invece di 70/15/15
  - grad_clip per stabilita
  - Atomic checkpoint save (os.replace)

Utilizzo:
  python train_pointer.py
"""

import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import chess

from MLChess import encode_legal_moves, JellyFishPointer
from encode_fast import encode_board_fast

MOVE_VECTOR_DIM = 46

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

CSV_FILE       = "stockfish_lichess_60m_fixed.csv"
CHECKPOINT_DIR = "checkpoints_pointer_60m_from_fast"
CHECKPOINT_OUT = os.path.join(CHECKPOINT_DIR, "last.pt")
CHECKPOINT_IN  = "checkpoints_bellman_fast/best.pt"   # es. "checkpoints_pointer_20m/best.pt" per riprendere

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True

EPOCHS        = 50
BATCH_SIZE    = 256
LR            = 5e-4
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0
POLICY_WEIGHT = 1.0
VALUE_WEIGHT  = 2.0
CP_SCALE      = 400
VAL_FRACTION  = 0.02
NUM_WORKERS   = 6

# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_value_std(v): return v.squeeze().float().std().item()


def encode_result(result_str: str, turn: bool) -> float:
    """
    Converte l'eval Stockfish (convenzione Lichess: sempre dal punto di
    vista del Bianco) in un target da -1 a +1 relativo al lato che deve
    muovere (side-to-move), coerente con la convenzione negamax usata
    dal value head / MCTS.
    """
    s = result_str.strip()
    if s.startswith('#'):
        val_white = 1.0 if s.startswith('#+') else -1.0
    else:
        try:    cp = float(s)
        except: cp = 0.0
        cp = max(-2000.0, min(2000.0, cp))
        val_white = math.tanh(cp / CP_SCALE)
    return val_white if turn == chess.WHITE else -val_white

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PointerDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """
        Usiamo array numpy a dtype fisso invece di tenere il DataFrame
        pandas come attributo. Motivo: con num_workers>0 il DataLoader fa
        fork dei worker, che ereditano il dataset via copy-on-write — ma
        il refcounting di Python tocca l'header di ogni oggetto ad ogni
        accesso (anche solo in lettura), e un DataFrame e' internamente
        migliaia di oggetti Python (stringhe, blocchi, indice). Questo
        rompe il COW e duplica fisicamente la memoria per ogni worker,
        moltiplicando la RAM usata per NUM_WORKERS. Un array numpy a dtype
        fisso (es. 'U100') e' un unico buffer contiguo: nessun refcounting
        per elemento, quindi il COW resta davvero condiviso tra i worker.
        """
        self.original_indices = df.index.to_numpy()
        # dtype 'S' (byte ASCII), non 'U' (unicode): 'U' usa 4 byte fissi
        # per carattere indipendentemente dal contenuto — per 60M righe,
        # 'U100' da solo costerebbe ~24GB solo per il FEN. FEN/UCI sono
        # puro ASCII, 'S' usa 1 byte/carattere: stessa capienza, 4x meno RAM.
        FEN_WIDTH, MOVE_WIDTH, EVAL_WIDTH = 100, 10, 20
        max_fen  = df["FEN"].str.len().max()
        max_move = df["Move"].str.len().max()
        max_eval = df["Evaluation"].astype(str).str.len().max()
        assert max_fen  <= FEN_WIDTH,  f"FEN piu' lungo del previsto: {max_fen} > {FEN_WIDTH}, alza FEN_WIDTH"
        assert max_move <= MOVE_WIDTH, f"Move piu' lungo del previsto: {max_move} > {MOVE_WIDTH}, alza MOVE_WIDTH"
        assert max_eval <= EVAL_WIDTH, f"Eval piu' lungo del previsto: {max_eval} > {EVAL_WIDTH}, alza EVAL_WIDTH"

        self.fens  = df["FEN"].to_numpy(dtype=f"S{FEN_WIDTH}")
        self.moves = df["Move"].to_numpy(dtype=f"S{MOVE_WIDTH}")
        self.evals = df["Evaluation"].astype(str).to_numpy(dtype=f"S{EVAL_WIDTH}")
        self._len  = len(df)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        fen      = self.fens[idx].decode("ascii")
        move_uci = self.moves[idx].decode("ascii")
        eval_str = self.evals[idx].decode("ascii")

        board      = chess.Board(fen)
        legal_list = list(board.legal_moves)

        board_t = encode_board_fast(fen)
        moves_t = encode_legal_moves(board)   # (N, 46)
        value   = encode_result(eval_str, board.turn)

        try:
            best  = chess.Move.from_uci(move_uci)
            label = legal_list.index(best) if best in legal_list else 0
        except:
            label = 0

        return board_t, moves_t, label, value


def collate_fn(batch):
    boards, moves_list, labels, values = zip(*batch)

    boards_t = torch.stack(boards)                          # (B, 13, 8, 8)
    moves_padded = pad_sequence(moves_list,
                                batch_first=True,
                                padding_value=0.0)          # (B, N_max, 46)

    N_max = moves_padded.shape[1]
    mask  = torch.zeros(len(batch), N_max, dtype=torch.bool)
    for i, m in enumerate(moves_list):
        mask[i, :len(m)] = True

    # One-hot policy target — evita NaN da CrossEntropyLoss con -inf nel padding
    policy_t = torch.zeros(len(batch), N_max)
    for i, lbl in enumerate(labels):
        policy_t[i, lbl] = 1.0

    labels_t = torch.tensor(labels, dtype=torch.long)
    values_t = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

    return boards_t, moves_padded, mask, policy_t, labels_t, values_t

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict() if scaler else None,
        "epoch":     epoch,
        "val_loss":  val_loss,
    }, tmp)
    os.replace(tmp, path)
    tqdm.write(f"  -> checkpoint salvato: {path}  (epoch {epoch}, val_loss {val_loss:.4f})")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location=DEVICE)
    sd   = ckpt.get("model", ckpt)
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    if optimizer and "optimizer" in ckpt:
        try: optimizer.load_state_dict(ckpt["optimizer"])
        except: pass
    if scheduler and "scheduler" in ckpt and ckpt["scheduler"]:
        try: scheduler.load_state_dict(ckpt["scheduler"])
        except: pass
    if scaler and "scaler" in ckpt and ckpt["scaler"]:
        try: scaler.load_state_dict(ckpt["scaler"])
        except: pass
    epoch    = ckpt.get("epoch", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    tqdm.write(f"  -> checkpoint caricato: {path}  (epoch {epoch})")
    return epoch, val_loss

# ---------------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, scaler, device, train=True):
    model.train() if train else model.eval()
    for p in model.parameters(): p.requires_grad = train

    total_p = total_v = total_acc = 0.0
    vstd_list = []
    n_batches = 0

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for boards, moves, mask, policy_t, labels, values in tqdm(loader, leave=False):
            boards   = boards.to(device)
            moves    = moves.to(device)
            mask     = mask.to(device)
            policy_t = policy_t.to(device)
            labels   = labels.to(device)
            values   = values.to(device)

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                _, probs, value_pred = model(boards, moves, mask)

                # Policy loss: cross-entropy manuale su probs mascherati
                # Evita NaN da log(0) usando probs gia' mascherati dal pointer head

                # Debug — rimuovi dopo
                log_probs = torch.log(probs + 1e-8)

                policy_loss = -(policy_t * log_probs).sum(dim=1).mean()

                if torch.isnan(policy_loss):
                    print(f"NaN rilevato!")
                    print(f"probs min: {probs.min().item():.6f}  max: {probs.max().item():.6f}")
                    print(f"NaN in probs: {torch.isnan(probs).any().item()}")
                    print(f"NaN in log_probs: {torch.isnan(log_probs).any().item()}")
                    print(f"Righe zero policy_t: {(policy_t.sum(dim=1) == 0).sum().item()}")
                    print(f"label max: {labels.max().item()}  N_max: {probs.shape[1]}")
                    raise RuntimeError("Stopping at NaN")

                value_loss = F.mse_loss(value_pred, values)
                loss       = POLICY_WEIGHT * policy_loss + VALUE_WEIGHT * value_loss

            if train:
                optimizer.zero_grad()
                if USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()

            with torch.no_grad():
                vstd_list.append(compute_value_std(value_pred))
                pred_idx = probs.argmax(dim=1)
                total_acc += (pred_idx == labels).float().mean().item()

            total_p += policy_loss.item()
            total_v += value_loss.item()
            n_batches += 1

    d = max(n_batches, 1)
    return {
        "policy_loss": total_p   / d,
        "value_loss":  total_v   / d,
        "accuracy":    total_acc / d,
        "value_std":   sum(vstd_list) / max(len(vstd_list), 1),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}  AMP: {USE_AMP}")

    # dtype espliciti: evita che pandas debba inferire i tipi (piu' lento e
    # con picchi di memoria maggiori durante il parsing di un CSV da 60M+ righe)
    df = pd.read_csv(CSV_FILE, dtype={"FEN": str, "Move": str, "Evaluation": str})
    df = df.dropna(subset=["FEN", "Move", "Evaluation"])
    df = df[df["Evaluation"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)
    print(f"Posizioni totali: {len(df):,}")

    val_size = max(1000, int(len(df) * VAL_FRACTION))
    val_df   = df.sample(n=val_size, random_state=42)
    train_df = df.drop(val_df.index)
    print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}\n")

    train_dataset = PointerDataset(train_df)
    val_dataset   = PointerDataset(val_df)

    # I DataFrame originali sono stati copiati in array numpy dentro i
    # Dataset — non servono piu'. Liberarli ORA, prima di creare i
    # DataLoader (quindi prima del fork dei worker), evita che restino
    # comunque in memoria come riferimenti pendenti nello scope di main().
    del df, train_df, val_df

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = JellyFishPointer().to(DEVICE)
    print(f"Parametri: {sum(p.numel() for p in model.parameters()):,}")

    # Carica checkpoint iniziale se specificato
    if CHECKPOINT_IN and os.path.exists(CHECKPOINT_IN):
        ckpt = torch.load(CHECKPOINT_IN, map_location=DEVICE)
        sd   = ckpt.get("model", ckpt)
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"Pesi caricati da: {CHECKPOINT_IN}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_val_loss = float("inf")
    start_epoch   = 1

    # Riprendi training se esiste last.pt
    if os.path.exists(CHECKPOINT_OUT):
        start_epoch, best_val_loss = load_checkpoint(
            CHECKPOINT_OUT, model, optimizer, scheduler, scaler
        )
        start_epoch += 1

    print(f"\nTraining da epoch {start_epoch} a {EPOCHS}\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        train_s = run_epoch(model, train_loader, optimizer, scaler, DEVICE, train=True)
        val_s   = run_epoch(model, val_loader,   optimizer, scaler, DEVICE, train=False)

        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train  p:{train_s['policy_loss']:.4f} v:{train_s['value_loss']:.4f} "
            f"acc:{train_s['accuracy']:.3f} vstd:{train_s['value_std']:.3f}  |  "
            f"val  p:{val_s['policy_loss']:.4f} v:{val_s['value_loss']:.4f} "
            f"acc:{val_s['accuracy']:.3f} vstd:{val_s['value_std']:.3f} | "
            f"LR:{optimizer.param_groups[0]['lr']:.1e}"
        )

        if train_s["value_std"] < 0.05:
            print(f"  ATTENZIONE possibile collasso value: vstd={train_s['value_std']:.4f}")

        gap = val_s["policy_loss"] - train_s["policy_loss"]
        if gap > 0.5:
            print(f"  ATTENZIONE overfitting: gap={gap:.3f}")

        val_loss = val_s["policy_loss"]
        save_checkpoint(model, optimizer, scheduler, scaler,
                        epoch, val_loss, CHECKPOINT_OUT)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, best_val_loss,
                            os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  * Nuovo best: {best_val_loss:.4f}")

    print(f"\nCompletato. Best val policy_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
