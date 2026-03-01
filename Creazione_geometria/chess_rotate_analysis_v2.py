"""
Analisi qualitativa profonda degli embedding ChessRotatE v2.

Domande a cui risponde questo script:
  1. La valutazione è codificata linearmente? (probing lineare)
  2. Le posizioni tatticamente simili si raggruppano? (UMAP)
  3. Le mosse hanno struttura geometrica? (clustering degli angoli)
  4. Esiste una "direzione del vantaggio" nello spazio? (PCA semantica)
  5. Le rotazioni delle mosse correlano col tipo di mossa?

Requisiti: pip install umap-learn scikit-learn matplotlib seaborn pandas torch chess tqdm
"""

import torch
import torch.nn.functional as F
import chess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from tqdm import tqdm
import sys

sys.path.append(".")
from chess_rotate_v2 import (
    ChessRotatEv2, ChessCSVDataset,
    board_to_tensor, move_to_features, parse_evaluation
)

# ── Palette colori coerente ──────────────────
C_POS  = "#2196F3"   # blu  = vantaggio bianco
C_NEG  = "#F44336"   # rosso = vantaggio nero
C_NEU  = "#9E9E9E"   # grigio = pari
C_ACC1 = "#FF9800"   # arancio
C_ACC2 = "#4CAF50"   # verde


# ════════════════════════════════════════════════════════
# UTILITY
# ════════════════════════════════════════════════════════

def load_model(path, embedding_dim=128, device="cpu"):
    model = ChessRotatEv2(embedding_dim)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    return model


def get_embedding_vec(model, board, device):
    """Ritorna il vettore concatenato [real, imag] shape [D*2]."""
    with torch.no_grad():
        t = board_to_tensor(board).unsqueeze(0).to(device)
        real, imag = model.position_encoder(t)
    return torch.cat([real, imag], dim=-1).squeeze(0).cpu().numpy()


def get_move_angles(model, board, move, device):
    """Ritorna gli angoli θ di rotazione per una mossa, shape [D]."""
    with torch.no_grad():
        feat = move_to_features(board, move).unsqueeze(0).to(device)
        theta = model.move_encoder(feat)
    return theta.squeeze(0).cpu().numpy()


def eval_to_color(eval_norm):
    """Mappa eval normalizzata [-1,1] a colore."""
    if eval_norm > 0.15:
        return C_POS
    elif eval_norm < -0.15:
        return C_NEG
    else:
        return C_NEU


# ════════════════════════════════════════════════════════
# 1. PROBING LINEARE
#    Quanto è lineare la codifica della valutazione?
#    Fitto una regressione Ridge sull'embedding → eval.
#    Se R² è alto, la valutazione è linearmente accessibile
#    dall'embedding — la info è "in superficie", non nascosta.
# ════════════════════════════════════════════════════════

def probing_analysis(model, dataset, device, n=3000):
    print("\n[1/5] Probing lineare della valutazione...")

    embeddings, evals = [], []
    indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)

    with torch.no_grad():
        for i in tqdm(indices, leave=False):
            pos_before, _, _, ev = dataset[i]
            t = pos_before.unsqueeze(0).to(device)
            real, imag = model.position_encoder(t)
            vec = torch.cat([real, imag], dim=-1).squeeze(0).cpu().numpy()
            embeddings.append(vec)
            evals.append(ev.item())

    X = np.array(embeddings)
    y = np.array(evals)

    # Split train/test
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # Regressione lineare
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_s, y_tr)
    y_pred = ridge.predict(X_te_s)
    r2_linear = r2_score(y_te, y_pred)

    print(f"  R² regressione lineare: {r2_linear:.4f}")
    print(f"  → La valutazione è {'linearmente accessibile' if r2_linear > 0.7 else 'non linearmente accessibile'} dall'embedding")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Probing lineare: la valutazione è lineare nell'embedding?",
                 fontsize=13, fontweight='bold')

    colors = [eval_to_color(e) for e in y_te]
    axes[0].scatter(y_te, y_pred, c=colors, alpha=0.4, s=10)
    axes[0].plot([-1, 1], [-1, 1], 'k--', linewidth=1.5)
    axes[0].set_xlabel("Eval reale")
    axes[0].set_ylabel("Eval predetta (probing lineare)")
    axes[0].set_title(f"R² = {r2_linear:.4f}")

    # Istogramma residui
    residuals = y_te - y_pred
    axes[1].hist(residuals, bins=50, color=C_ACC1, alpha=0.8, edgecolor='white')
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].set_xlabel("Residuo (reale - predetto)")
    axes[1].set_title(f"Distribuzione residui (std={residuals.std():.3f})")

    plt.tight_layout()
    plt.savefig("probing_lineare.png", dpi=150, bbox_inches='tight')
    print("  Salvato: probing_lineare.png")
    plt.close()

    return r2_linear, np.array(embeddings), np.array(evals)


# ════════════════════════════════════════════════════════
# 2. DIREZIONE DEL VANTAGGIO
#    Trova la direzione principale di variazione legata
#    alla valutazione usando PCA pesata.
#    Poi proietta tutti i punti su questa direzione.
# ════════════════════════════════════════════════════════

def advantage_direction(embeddings, evals):
    print("\n[2/5] Analisi della direzione del vantaggio...")

    # PCA globale
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    pca = PCA(n_components=10)
    coords = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    # Correlazione di ogni PC con la valutazione
    correlations = []
    for i in range(10):
        corr = np.corrcoef(coords[:, i], evals)[0, 1]
        correlations.append(corr)
        print(f"  PC{i+1} ({explained[i]:.1%} var) — corr con eval: {corr:+.4f}")

    best_pc = np.argmax(np.abs(correlations))
    print(f"\n  → PC{best_pc+1} è la più correlata con la valutazione (r={correlations[best_pc]:+.4f})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Direzione del vantaggio nello spazio latente",
                 fontsize=13, fontweight='bold')

    # PC1 vs PC2, colorato per eval
    scatter = axes[0].scatter(
        coords[:, 0], coords[:, 1],
        c=evals, cmap='RdBu', vmin=-1, vmax=1,
        alpha=0.4, s=8
    )
    plt.colorbar(scatter, ax=axes[0], label="Valutazione normalizzata")
    axes[0].set_xlabel(f"PC1 ({explained[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({explained[1]:.1%})")
    axes[0].set_title("PCA colorato per valutazione\n(rosso=nero, blu=bianco)")

    # Proiezione sulla PC più correlata con eval
    proj = coords[:, best_pc]
    # Scatter: proiezione vs eval reale
    colors = [eval_to_color(e) for e in evals]
    axes[1].scatter(proj, evals, c=colors, alpha=0.3, s=8)
    axes[1].set_xlabel(f"Proiezione su PC{best_pc+1}")
    axes[1].set_ylabel("Valutazione reale")
    axes[1].set_title(f"La 'direzione del vantaggio'\nr={correlations[best_pc]:+.4f}")

    plt.tight_layout()
    plt.savefig("direzione_vantaggio.png", dpi=150, bbox_inches='tight')
    print("  Salvato: direzione_vantaggio.png")
    plt.close()

    return coords, explained, correlations


# ════════════════════════════════════════════════════════
# 3. UMAP — CLUSTERING POSIZIONI
#    Riduce a 2D con UMAP (preserva struttura locale)
#    e visualizza per fase, eval, tipo posizione.
# ════════════════════════════════════════════════════════

def umap_analysis(embeddings, evals, dataset, indices):
    print("\n[3/5] UMAP delle posizioni...")

    try:
        import umap
    except ImportError:
        print("  UMAP non installato. Esegui: pip install umap-learn")
        return

    # Metadati
    phases, n_pieces_list = [], []
    for i in indices[:len(embeddings)]:
        pos_before, _, _, _ = dataset[i]
        # Ricostruisci board dal tensore (conta pezzi)
        n_pieces = int(pos_before.sum().item())
        n_pieces_list.append(n_pieces)
        if n_pieces >= 28:
            phases.append("apertura")
        elif n_pieces >= 14:
            phases.append("mediogioco")
        else:
            phases.append("finale")

    print("  Fitting UMAP (può richiedere 1-2 minuti)...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        random_state=42, verbose=False)
    coords_2d = reducer.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("UMAP degli embedding di posizione", fontsize=13, fontweight='bold')

    # Plot 1: colorato per valutazione
    scatter = axes[0].scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=evals, cmap='RdBu', vmin=-1, vmax=1,
        alpha=0.4, s=6
    )
    plt.colorbar(scatter, ax=axes[0], label="Eval")
    axes[0].set_title("Colorato per valutazione")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Plot 2: colorato per fase
    phase_color_map = {"apertura": C_POS, "mediogioco": C_ACC1, "finale": C_ACC2}
    phase_colors = [phase_color_map[p] for p in phases]
    for phase, col in phase_color_map.items():
        mask = [p == phase for p in phases]
        axes[1].scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=col, label=phase, alpha=0.4, s=6
        )
    axes[1].set_title("Colorato per fase di gioco")
    axes[1].legend(markerscale=3)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # Plot 3: colorato per numero di pezzi
    scatter3 = axes[2].scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=n_pieces_list, cmap='viridis',
        alpha=0.4, s=6
    )
    plt.colorbar(scatter3, ax=axes[2], label="N pezzi")
    axes[2].set_title("Colorato per numero di pezzi")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    plt.tight_layout()
    plt.savefig("umap_posizioni.png", dpi=150, bbox_inches='tight')
    print("  Salvato: umap_posizioni.png")
    plt.close()


# ════════════════════════════════════════════════════════
# 4. STRUTTURA GEOMETRICA DELLE MOSSE
#    Estrai gli angoli θ per ~2000 mosse.
#    Clustera per tipo e visualizza.
#    Domanda: mosse simili (stesso pezzo, stessa direzione)
#    hanno angoli simili nello spazio?
# ════════════════════════════════════════════════════════

MOVE_TYPES = {
    "pedone_avanza": lambda b, m: (
        b.piece_at(m.from_square) and
        b.piece_at(m.from_square).piece_type == chess.PAWN and
        not b.is_capture(m)
    ),
    "pedone_cattura": lambda b, m: (
        b.piece_at(m.from_square) and
        b.piece_at(m.from_square).piece_type == chess.PAWN and
        b.is_capture(m)
    ),
    "cavallo": lambda b, m: (
        b.piece_at(m.from_square) and
        b.piece_at(m.from_square).piece_type == chess.KNIGHT
    ),
    "alfiere": lambda b, m: (
        b.piece_at(m.from_square) and
        b.piece_at(m.from_square).piece_type == chess.BISHOP
    ),
    "torre": lambda b, m: (
        b.piece_at(m.from_square) and
        b.piece_at(m.from_square).piece_type == chess.ROOK
    ),
    "donna": lambda b, m: (
        b.piece_at(m.from_square) and
        b.piece_at(m.from_square).piece_type == chess.QUEEN
    ),
    "re": lambda b, m: (
        b.piece_at(m.from_square) and
        b.piece_at(m.from_square).piece_type == chess.KING
    ),
}

MOVE_COLORS = {
    "pedone_avanza":  "#795548",
    "pedone_cattura": "#FF5722",
    "cavallo":        "#9C27B0",
    "alfiere":        "#2196F3",
    "torre":          "#009688",
    "donna":          "#F44336",
    "re":             "#FF9800",
}


def move_geometry_analysis(model, dataset, device, n=2000):
    print("\n[4/5] Struttura geometrica delle mosse...")

    all_angles = []
    all_types = []
    all_is_capture = []

    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), min(n * 3, len(dataset)), replace=False)

    with torch.no_grad():
        for i in indices:
            if len(all_angles) >= n:
                break
            pos_before, move_feat, _, _ = dataset[i]

            # Ottieni angoli
            feat = move_feat.unsqueeze(0).to(device)
            theta = model.move_encoder(feat).squeeze(0).cpu().numpy()

            # Determina tipo mossa dalle feature
            # feat[10] = is_capture, feat[4:10] = tipo pezzo (one-hot)
            piece_idx = int(move_feat[4:10].argmax().item())
            piece_names = ["pedone", "cavallo", "alfiere", "torre", "donna", "re"]
            is_capture = move_feat[10].item() > 0.5
            piece_name = piece_names[piece_idx]

            if piece_name == "pedone":
                move_type = "pedone_cattura" if is_capture else "pedone_avanza"
            else:
                move_type = piece_name

            all_angles.append(theta)
            all_types.append(move_type)
            all_is_capture.append(is_capture)

    all_angles = np.array(all_angles)  # [N, D]
    print(f"  Mosse analizzate: {len(all_angles)}")

    # Distribuzione per tipo
    type_counts = defaultdict(int)
    for t in all_types:
        type_counts[t] += 1
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:20s}: {c}")

    # PCA degli angoli
    scaler = StandardScaler()
    angles_scaled = scaler.fit_transform(all_angles)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(angles_scaled)
    explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Geometria delle mosse nello spazio degli angoli di rotazione",
                 fontsize=13, fontweight='bold')

    # Plot 1: PCA degli angoli, colorato per tipo pezzo
    for mtype in MOVE_TYPES.keys():
        mask = [t == mtype for t in all_types]
        if sum(mask) > 0:
            axes[0].scatter(
                coords[mask, 0], coords[mask, 1],
                c=MOVE_COLORS[mtype], label=mtype,
                alpha=0.5, s=15
            )
    axes[0].set_xlabel(f"PC1 ({explained[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({explained[1]:.1%})")
    axes[0].set_title("Tipo di pezzo mosso")
    axes[0].legend(markerscale=2, fontsize=8)

    # Plot 2: cattura vs non-cattura
    cap_colors = [C_NEG if c else C_POS for c in all_is_capture]
    axes[1].scatter(coords[:, 0], coords[:, 1],
                    c=cap_colors, alpha=0.4, s=10)
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=C_NEG, label='Cattura'),
                  Patch(facecolor=C_POS, label='Mossa tranquilla')]
    axes[1].legend(handles=legend_els)
    axes[1].set_xlabel(f"PC1 ({explained[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({explained[1]:.1%})")
    axes[1].set_title("Cattura vs mossa tranquilla")

    plt.tight_layout()
    plt.savefig("geometria_mosse.png", dpi=150, bbox_inches='tight')
    print("  Salvato: geometria_mosse.png")
    plt.close()

    # Distanze medie tra tipi di mossa
    print("\n  Distanze medie tra centroidi dei tipi di mossa:")
    centroids = {}
    for mtype in MOVE_TYPES.keys():
        mask = np.array([t == mtype for t in all_types])
        if mask.sum() > 10:
            centroids[mtype] = all_angles[mask].mean(axis=0)

    types_list = list(centroids.keys())
    print(f"  {'':20s}", end="")
    for t in types_list:
        print(f"  {t[:8]:>8s}", end="")
    print()
    for t1 in types_list:
        print(f"  {t1:20s}", end="")
        for t2 in types_list:
            dist = np.linalg.norm(centroids[t1] - centroids[t2])
            print(f"  {dist:8.2f}", end="")
        print()

    return all_angles, all_types


# ════════════════════════════════════════════════════════
# 5. TEST POSIZIONI SPECIFICHE
#    Posizioni scacchisticamente note — dove finiscono?
#    Sono dove ci aspettiamo?
# ════════════════════════════════════════════════════════

KNOWN_POSITIONS = {
    # Posizione iniziale
    "Iniziale":
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",

    # Finale Re+Donna vs Re (vantaggio decisivo bianco)
    "KQ vs K (bianco vince)":
        "8/8/8/8/8/3k4/8/3KQ3 w - - 0 1",

    # Finale Re+Pedone vs Re (vantaggio lieve)
    "KP vs K (bianco lieve)":
        "8/8/8/8/8/3k4/4P3/3K4 w - - 0 1",

    # Finale Re vs Re (patta)
    "K vs K (patta)":
        "8/8/8/8/8/3k4/8/3K4 w - - 0 1",

    # Posizione con vantaggio netto nero
    "Nero domina (torre extra)":
        "8/8/8/8/8/3k4/8/3Kr3 w - - 0 1",

    # Posizione tatticamente tesa
    "Posizione aperta":
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",

    # Fine partita classica
    "Finale di torre":
        "8/pp6/2p5/8/8/2P5/PP6/4K2R w - - 0 1",
}

def known_positions_analysis(model, device, coords_pca, embeddings_pca, evals_pca):
    print("\n[5/5] Analisi posizioni note...")

    known_vecs = {}
    known_preds = {}

    with torch.no_grad():
        for name, fen in KNOWN_POSITIONS.items():
            try:
                board = chess.Board(fen)
                t = board_to_tensor(board).unsqueeze(0).to(device)
                real, imag = model.position_encoder(t)
                pred_eval = model.eval_head(real, imag).item()
                vec = torch.cat([real, imag], dim=-1).squeeze(0).cpu().numpy()
                known_vecs[name] = vec
                known_preds[name] = pred_eval
                print(f"  {name:35s} eval predetta: {pred_eval:+.4f}")
            except Exception as e:
                print(f"  {name}: errore — {e}")

    # Proietta le posizioni note sullo stesso spazio PCA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(embeddings_pca)
    pca = PCA(n_components=2)
    pca.fit(scaler.transform(embeddings_pca))

    fig, ax = plt.subplots(figsize=(10, 8))

    # Sfondo: tutte le posizioni del dataset
    scatter = ax.scatter(
        coords_pca[:, 0], coords_pca[:, 1],
        c=evals_pca, cmap='RdBu', vmin=-1, vmax=1,
        alpha=0.15, s=5, zorder=1
    )
    plt.colorbar(scatter, ax=ax, label="Eval dataset")

    # Overlay: posizioni note
    for name, vec in known_vecs.items():
        vec_s = scaler.transform(vec.reshape(1, -1))
        coord = pca.transform(vec_s)[0]
        pred = known_preds[name]
        color = eval_to_color(pred)
        ax.scatter(coord[0], coord[1], c=color, s=200,
                   edgecolors='black', linewidths=1.5, zorder=3)
        ax.annotate(
            f"{name}\n({pred:+.2f})",
            (coord[0], coord[1]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=7, zorder=4,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
        )

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Posizioni note proiettate nello spazio PCA\n"
                 "(rosso=vantaggio nero, blu=vantaggio bianco)",
                 fontsize=12)

    plt.tight_layout()
    plt.savefig("posizioni_note.png", dpi=150, bbox_inches='tight')
    print("  Salvato: posizioni_note.png")
    plt.close()


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    CSV_PATH   = sys.argv[1] if len(sys.argv) > 1 else "over_mate_1_tactic_evals.csv"
    MODEL_PATH = sys.argv[2] if len(sys.argv) > 2 else "chess_rotate_v2.pt"
    N_SAMPLES  = int(sys.argv[3]) if len(sys.argv) > 3 else 3000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Campioni: {N_SAMPLES}")

    model = load_model(MODEL_PATH, embedding_dim=128, device=device)

    # Dataset
    dataset = ChessCSVDataset(CSV_PATH, max_samples=N_SAMPLES * 2)

    # Analisi 1: probing lineare + estrazione embedding
    r2_lin, embeddings, evals = probing_analysis(model, dataset, device, n=N_SAMPLES)
    indices = np.random.choice(len(dataset), min(N_SAMPLES, len(dataset)), replace=False)

    # Analisi 2: direzione del vantaggio
    coords_pca, explained, correlations = advantage_direction(embeddings, evals)

    # Analisi 3: UMAP
    umap_analysis(embeddings, evals, dataset, indices)

    # Analisi 4: geometria mosse
    all_angles, all_types = move_geometry_analysis(model, dataset, device, n=2000)

    # Analisi 5: posizioni note
    known_positions_analysis(model, device, coords_pca, embeddings, evals)

    print("\n════════════════════════════════════════")
    print("SOMMARIO")
    print("════════════════════════════════════════")
    print(f"R² probing lineare:          {r2_lin:.4f}")
    best_pc = int(np.argmax(np.abs(correlations)))
    print(f"PC più correlata con eval:   PC{best_pc+1} (r={correlations[best_pc]:+.4f})")
    print(f"Varianza spiegata PC1+PC2:   {explained[0]+explained[1]:.1%}")
    print()
    print("File generati:")
    for f in ["probing_lineare.png", "direzione_vantaggio.png",
              "umap_posizioni.png", "geometria_mosse.png", "posizioni_note.png"]:
        print(f"  {f}")
