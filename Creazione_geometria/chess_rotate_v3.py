"""
Chess RotatE v2 — con dataset reale (FEN + Evaluation + Move)
e loss supervisionata sulla valutazione.

Due loss combinate:
  1. RotatE loss:    h ⊙ e^(iθ) ≈ t  (coerenza geometrica delle mosse)
  2. Eval loss:      posizioni simili in valutazione → vicine nello spazio

Requisiti:
    pip install torch chess numpy tqdm pandas scikit-learn matplotlib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re


# ─────────────────────────────────────────────
# 1. PARSING DELLA VALUTAZIONE
# ─────────────────────────────────────────────

def parse_evaluation(eval_str: str) -> float:
    """
    Converte la stringa di valutazione in un float normalizzato in [-1, 1].

    Esempi:
      "#+2"  →  +1.0   (matto bianco)
      "#-3"  →  -1.0   (matto nero)
      "+408" →  +0.408 (vantaggio bianco, ~4 pedoni)
      "-150" →  -0.15  (vantaggio nero, ~1.5 pedoni)
      "+444" →  +0.444
    """
    eval_str = str(eval_str).strip()

    # Matto
    if eval_str.startswith("#+") or eval_str == "#":
        return 1.0
    if eval_str.startswith("#-"):
        return -1.0
    if eval_str.startswith("#"):
        # "#2" senza segno = matto bianco
        return 1.0

    # Numerico: rimuovi "+" e converti
    try:
        val = float(eval_str.replace("+", ""))
        # Normalizza: 1000 centipawn = vantaggio decisivo
        # tanh schiaccia i valori estremi in [-1, 1]
        return float(np.tanh(val / 600.0))  # calibrato sul 90° percentile del dataset
    except ValueError:
        return 0.0  # fallback


# ─────────────────────────────────────────────
# 2. RAPPRESENTAZIONE POSIZIONE E MOSSA
# ─────────────────────────────────────────────

PIECE_TO_LAYER = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

PIECE_TYPE_IDX = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4,  chess.KING: 5
}
PROMO_IDX = {
    chess.KNIGHT: 0, chess.BISHOP: 1,
    chess.ROOK: 2,   chess.QUEEN: 3
}

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros(12, 8, 8)
    for square, piece in board.piece_map().items():
        layer = PIECE_TO_LAYER[(piece.piece_type, piece.color)]
        tensor[layer, square // 8, square % 8] = 1.0
    return tensor

def move_to_features(board: chess.Board, move: chess.Move) -> torch.Tensor:
    feat = torch.zeros(16)
    feat[0] = (move.from_square % 8) / 7.0
    feat[1] = (move.from_square // 8) / 7.0
    feat[2] = (move.to_square % 8) / 7.0
    feat[3] = (move.to_square // 8) / 7.0
    piece = board.piece_at(move.from_square)
    if piece is not None:
        feat[4 + PIECE_TYPE_IDX[piece.piece_type]] = 1.0
    feat[10] = 1.0 if board.is_capture(move) else 0.0
    if move.promotion is not None:
        feat[11] = 1.0
        feat[12 + PROMO_IDX[move.promotion]] = 1.0
    return feat


# ─────────────────────────────────────────────
# 3. DATASET DAL CSV REALE
# ─────────────────────────────────────────────

class ChessCSVDataset(Dataset):
    """
    Carica il CSV con colonne: FEN, Evaluation, Move
    
    Ogni sample è una tripla:
      (pos_before_tensor, move_features, pos_after_tensor, eval_before, eval_after)
    
    eval_before: valutazione della posizione di partenza
    eval_after:  valutazione della posizione dopo la mossa
                 (approssimata come -eval_before * decay, o da una seconda lookup)
    """

    def __init__(
        self,
        csv_path: str,
        max_samples: int = 200_000,
        skip_invalid: bool = True,
    ):
        self.samples = []
        self._load(csv_path, max_samples, skip_invalid)

    def _load(self, csv_path: str, max_samples: int, skip_invalid: bool):
        print(f"Caricando {csv_path}...")
        df = pd.read_csv(csv_path, nrows=max_samples)
        print(f"Righe lette: {len(df)}")

        # Normalizza nomi colonne (case-insensitive)
        df.columns = [c.strip().lower() for c in df.columns]

        skipped = 0
        print("Processando posizioni...")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                fen = str(row["fen"]).strip()
                eval_str = str(row["evaluation"]).strip()
                move_str = str(row["move"]).strip()

                # Parse board
                board = chess.Board(fen)

                # Parse mossa (formato UCI: e.g. "d3g6", "h3g4")
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    if skip_invalid:
                        skipped += 1
                        continue

                # Valutazione posizione di partenza
                eval_before = parse_evaluation(eval_str)

                # Tensori
                pos_before = board_to_tensor(board)
                move_feat = move_to_features(board, move)

                # Posizione dopo la mossa
                board.push(move)
                pos_after = board_to_tensor(board)

                self.samples.append((
                    pos_before,
                    move_feat,
                    pos_after,
                    torch.tensor([eval_before], dtype=torch.float32),
                ))

            except Exception:
                skipped += 1
                continue

        print(f"Sample validi: {len(self.samples)} | Saltati: {skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Ritorna: (pos_before, move_feat, pos_after, eval_before)
        return self.samples[idx]


# ─────────────────────────────────────────────
# 4. ENCODER POSIZIONE (CNN)
# ─────────────────────────────────────────────

class PositionEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.conv = nn.Sequential(
            nn.Conv2d(12, 64,  kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim * 2)
        )

    def forward(self, x):
        features = self.conv(x)
        out = self.fc(features)
        real = out[:, :self.embedding_dim]
        imag = out[:, self.embedding_dim:]
        return real, imag


# ─────────────────────────────────────────────
# 5. ENCODER MOSSA (MLP)
# ─────────────────────────────────────────────

class MoveEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, feature_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, move_features):
        return self.mlp(move_features)


# ─────────────────────────────────────────────
# 6. TESTA DI VALUTAZIONE
#    Predice la valutazione dalla norma + direzione
#    dell'embedding. Piccola, non domina il training.
# ─────────────────────────────────────────────

class EvalHead(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, real, imag):
        x = torch.cat([real, imag], dim=-1)
        return self.head(x)


# ─────────────────────────────────────────────
# 7. MODELLO COMPLETO
# ─────────────────────────────────────────────

class ChessRotatEv4(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.position_encoder = PositionEncoder(embedding_dim)
        self.move_encoder = MoveEncoder(embedding_dim)
        self.eval_head = EvalHead(embedding_dim)

    def rotate(self, h_real, h_imag, theta):
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        rotated_real = h_real * cos_t - h_imag * sin_t
        rotated_imag = h_real * sin_t + h_imag * cos_t
        return rotated_real, rotated_imag

    def forward(self, pos_before, move_feat, pos_after):
        h_real, h_imag = self.position_encoder(pos_before)
        t_real, t_imag = self.position_encoder(pos_after)

        theta = self.move_encoder(move_feat)
        pred_real, pred_imag = self.rotate(h_real, h_imag, theta)

        diff_real = pred_real - t_real
        diff_imag = pred_imag - t_imag
        rotate_dist = torch.sqrt(diff_real**2 + diff_imag**2 + 1e-8).sum(dim=-1)

        eval_pred_before = self.eval_head(h_real, h_imag)

        return {
            "rotate_dist":      rotate_dist,
            "h_real": h_real,   "h_imag": h_imag,
            "t_real": t_real,   "t_imag": t_imag,
            "eval_pred_before": eval_pred_before,
        }


# ─────────────────────────────────────────────
# 8. LOSS COMBINATA
#
#  L_total = λ1 * L_rotate + λ2 * L_eval + λ3 * L_contrastive
#
#  L_rotate:     margin loss tra triple positive e negative
#  L_eval:       MSE tra valutazione predetta e reale
#  L_contrastive: coppie casuali nel batch — se eval simile → vicine,
#                 se eval diversa → lontane. NON usa eval_after approssimata.
# ─────────────────────────────────────────────

class CombinedLoss(nn.Module):
    def __init__(
        self,
        margin: float = 6.0,
        lambda_rotate: float = 1.0,
        lambda_eval: float = 2.0,
        lambda_contrastive: float = 0.5,
        contrastive_margin: float = 0.5,
    ):
        super().__init__()
        self.margin = margin
        self.lambda_rotate = lambda_rotate
        self.lambda_eval = lambda_eval
        self.lambda_contrastive = lambda_contrastive
        self.contrastive_margin = contrastive_margin

    def rotate_loss(self, pos_dist, neg_dist):
        return F.relu(self.margin + pos_dist - neg_dist).mean()

    def eval_loss(self, pred, target):
        return F.mse_loss(pred.squeeze(-1), target.squeeze(-1))

    def contrastive_loss(self, real_a, imag_a, eval_a,
                               real_b, imag_b, eval_b):
        """
        Per coppie di posizioni (a, b) nel batch:
        - Se |eval_a - eval_b| < soglia → sono "simili" → distanza spaziale piccola
        - Se |eval_a - eval_b| > soglia → sono "diverse" → distanza spaziale grande

        Usa una margin contrastive loss standard:
          simili:  loss = dist²
          diverse: loss = max(0, margin - dist)²
        """
        # Distanza euclidea nello spazio latente
        diff_r = real_a - real_b
        diff_i = imag_a - imag_b
        dist = torch.sqrt((diff_r**2 + diff_i**2).sum(dim=-1) + 1e-8)
        # Normalizza per rendere la distanza confrontabile col margin
        dist_norm = dist / (dist.detach().mean() + 1e-8)

        # Differenza di valutazione
        eval_diff = torch.abs(eval_a.squeeze(-1) - eval_b.squeeze(-1))

        # Soglia: 0.2 su scala [-1,1] ≈ 220 centipawn
        similar = (eval_diff < 0.2).float()
        different = 1.0 - similar

        loss_similar   = similar   * dist_norm**2
        loss_different = different * F.relu(self.contrastive_margin - dist_norm)**2

        return (loss_similar + loss_different).mean()

    def forward(self, out_pos, out_neg, eval_batch):
        """
        eval_batch: [batch] valutazioni reali delle posizioni di partenza
        Per la contrastive loss usiamo coppie (i, i+batch//2) nel batch stesso.
        """
        l_rot = self.rotate_loss(
            out_pos["rotate_dist"],
            out_neg["rotate_dist"]
        )

        l_eval = self.eval_loss(
            out_pos["eval_pred_before"],
            eval_batch
        )

        # Costruisci coppie: prima metà del batch vs seconda metà
        batch_size = out_pos["h_real"].shape[0]
        half = batch_size // 2
        if half > 0:
            l_con = self.contrastive_loss(
                out_pos["h_real"][:half], out_pos["h_imag"][:half],
                eval_batch[:half],
                out_pos["h_real"][half:half*2], out_pos["h_imag"][half:half*2],
                eval_batch[half:half*2],
            )
        else:
            l_con = torch.tensor(0.0, device=eval_batch.device)

        total = (self.lambda_rotate      * l_rot +
                 self.lambda_eval        * l_eval +
                 self.lambda_contrastive * l_con)

        return total, {
            "rotate":      l_rot.item(),
            "eval":        l_eval.item(),
            "contrastive": l_con.item(),
        }


# ─────────────────────────────────────────────
# 9. NEGATIVE SAMPLING
# ─────────────────────────────────────────────

def corrupt_batch(pos_after_batch, dataset, device):
    """Sostituisce pos_after con posizioni casuali dal dataset."""
    batch_size = pos_after_batch.shape[0]
    indices = torch.randint(0, len(dataset), (batch_size,))
    corrupted = torch.stack([dataset[i][2] for i in indices])
    return corrupted.to(device)


# ─────────────────────────────────────────────
# 10. TRAINING
# ─────────────────────────────────────────────

def train(
    csv_path: str,
    embedding_dim: int = 128,
    max_samples: int = 200_000,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    lambda_rotate: float = 1.0,
    lambda_eval: float = 0.5,
    lambda_semantic: float = 0.3,
    save_path: str = "chess_rotate_v3.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Device: {device} | Embedding dim: {embedding_dim}")

    dataset = ChessCSVDataset(csv_path, max_samples=max_samples)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(device == "cuda")
    )

    model = ChessRotatEv4(embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    loss_fn = CombinedLoss(
        lambda_rotate=lambda_rotate,
        lambda_eval=lambda_eval,
        lambda_contrastive=lambda_semantic,
    )

    print(f"Parametri: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch per epoch: {len(loader)}\n")

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"total": 0.0, "rotate": 0.0, "eval": 0.0, "contrastive": 0.0}

        for batch in tqdm(loader, desc=f"Epoch {epoch:2d}/{epochs}", leave=False):
            pos_before, move_feat, pos_after, eval_before = batch
            pos_before  = pos_before.to(device)
            move_feat   = move_feat.to(device)
            pos_after   = pos_after.to(device)
            eval_before = eval_before.to(device)

            out_pos = model(pos_before, move_feat, pos_after)

            pos_after_neg = corrupt_batch(pos_after, dataset, device)
            out_neg = model(pos_before, move_feat, pos_after_neg)

            loss, components = loss_fn(out_pos, out_neg, eval_before)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            totals["total"]       += loss.item()
            totals["rotate"]      += components["rotate"]
            totals["eval"]        += components["eval"]
            totals["contrastive"] += components["contrastive"]

        scheduler.step()
        n = len(loader)
        log = {k: v / n for k, v in totals.items()}
        history.append(log)

        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Total: {log['total']:.4f} | "
              f"RotatE: {log['rotate']:.4f} | "
              f"Eval: {log['eval']:.4f} | "
              f"Contrastive: {log['contrastive']:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "embedding_dim": embedding_dim,
        "history": history,
    }, save_path)
    print(f"\nModello salvato in {save_path}")
    return model, history


# ─────────────────────────────────────────────
# 11. ANALISI POST-TRAINING
# ─────────────────────────────────────────────

def analyze(model, csv_path, n_samples=2000,
            device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Analisi rapida post-training:
    - Correlazione tra eval reale e eval predetta
    - Correlazione tra norma embedding e valutazione
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    dataset = ChessCSVDataset(csv_path, max_samples=n_samples)
    model.eval()
    model.to(device)

    eval_real = []
    eval_pred = []
    norms = []

    with torch.no_grad():
        for pos_before, _, _, ev_before in tqdm(dataset, desc="Analisi"):
            t = pos_before.unsqueeze(0).to(device)
            real, imag = model.position_encoder(t)
            pred = model.eval_head(real, imag).item()
            norm = torch.sqrt(real**2 + imag**2).mean().item()
            eval_real.append(ev_before.item())
            eval_pred.append(pred)
            norms.append(norm)

    eval_real = np.array(eval_real)
    eval_pred = np.array(eval_pred)
    norms = np.array(norms)

    r2 = r2_score(eval_real, eval_pred)
    corr_norm = np.corrcoef(norms, np.abs(eval_real))[0, 1]

    print(f"\nR² eval predetta vs reale: {r2:.4f}")
    print(f"Correlazione norma vs |eval|: {corr_norm:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Analisi post-training v2", fontsize=14, fontweight='bold')

    axes[0].scatter(eval_real, eval_pred, alpha=0.2, s=5, color="#2196F3")
    axes[0].plot([-1, 1], [-1, 1], 'r--', linewidth=2)
    axes[0].set_xlabel("Valutazione reale (normalizzata)")
    axes[0].set_ylabel("Valutazione predetta")
    axes[0].set_title(f"Eval reale vs predetta (R²={r2:.3f})")

    axes[1].scatter(np.abs(eval_real), norms, alpha=0.2, s=5, color="#9C27B0")
    axes[1].set_xlabel("|Valutazione| (vantaggio assoluto)")
    axes[1].set_ylabel("Norma embedding")
    axes[1].set_title(f"Norma vs |Eval| (r={corr_norm:.3f})")

    plt.tight_layout()
    plt.savefig("analisi_v3.png", dpi=150, bbox_inches='tight')
    print("Salvato: analisi_v3.png")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "chess_data.csv"

    model, history = train(
        csv_path=CSV_PATH,
        embedding_dim=128,
        max_samples=200_000,
        epochs=20,
        batch_size=128,
        lr=1e-3,
        lambda_rotate=1.0,
        lambda_eval=2.0,
        lambda_semantic=0.5,
    )

    analyze(model, CSV_PATH, n_samples=5000)
