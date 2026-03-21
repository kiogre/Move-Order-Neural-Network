"""
visualize.py
------------
Visualization tools for chess influence fields.
Requires: matplotlib, chess (python-chess)

Install:  pip install matplotlib python-chess
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as mpe
from matplotlib.patches import Rectangle
from chess_fields import (
    compute_fields_from_fen, fen_to_board, PIECE_SYMBOLS, EMPTY
)


# ---------------------------------------------------------------------------
# Board drawing helpers
# ---------------------------------------------------------------------------

LIGHT_SQ = '#F0D9B5'
DARK_SQ  = '#B58863'


def _draw_board_base(ax, board: np.ndarray):
    """Draw the chessboard squares and piece symbols."""
    for r in range(8):
        for c in range(8):
            color = LIGHT_SQ if (r + c) % 2 == 0 else DARK_SQ
            ax.add_patch(Rectangle((c, 7-r), 1, 1, color=color, zorder=0))

            piece = board[r, c]
            if piece != EMPTY:
                symbol = PIECE_SYMBOLS[piece]
                txt_color = 'white' if piece < 0 else 'black'
                ax.text(
                    c + 0.5, 7 - r + 0.5, symbol,
                    ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color=txt_color, zorder=3,
                    path_effects=[
                        mpe.withStroke(
                            linewidth=2,
                            foreground='black' if piece < 0 else 'white'
                        )
                    ]
                )

    # Rank/file labels
    for i in range(8):
        ax.text(i + 0.5, -0.25, 'abcdefgh'[i], ha='center', va='center', fontsize=9)
        ax.text(-0.25, i + 0.5, str(i+1), ha='center', va='center', fontsize=9)

    ax.set_xlim(-0.35, 8.1)
    ax.set_ylim(-0.35, 8.1)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_control_field(
    fen: str,
    alpha: float = 0.5,
    figsize: tuple = (7, 7),
    cmap: str = 'RdBu',
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plot the control field C(x) = white_influence - black_influence
    overlaid on the chessboard.

    Blue  = white controls
    Red   = black controls
    White = contested
    """
    fields = compute_fields_from_fen(fen, alpha=alpha)
    board  = fields['board']
    C      = fields['control']

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Overlay heatmap (flip rows: row0=rank8, displayed at top)
    C_display = C[::-1, :]
    vmax = max(abs(C_display).max(), 1e-6)
    im = ax.imshow(
        C_display, cmap=cmap,
        extent=[0, 8, 0, 8],
        vmin=-vmax, vmax=vmax,
        alpha=0.55, zorder=1,
        interpolation='nearest'
    )

    _draw_board_base(ax, board)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='White ← 0 → Black')

    if title is None:
        title = f'Control Field  (α={alpha})'
    ax.set_title(title, fontsize=12)

    fig.tight_layout()
    return fig


def plot_piece_influence(
    fen: str,
    piece_square: tuple[int, int],
    alpha: float = 0.5,
    figsize: tuple = (6, 6),
) -> plt.Figure:
    """
    Plot the influence field of a single piece at (row, col).
    """
    from chess_fields import bfs_distances, compute_piece_influence, fen_to_board

    board = fen_to_board(fen)
    r, c  = piece_square
    piece = board[r, c]

    if piece == EMPTY:
        raise ValueError(f"No piece at ({r},{c})")

    dist = bfs_distances(board, r, c, piece)
    phi  = compute_piece_influence(dist, alpha)

    fig, ax = plt.subplots(figsize=figsize)

    phi_display = phi[::-1, :]
    im = ax.imshow(
        phi_display, cmap='YlOrRd',
        extent=[0, 8, 0, 8],
        vmin=0, vmax=1,
        alpha=0.6, zorder=1,
        interpolation='nearest'
    )
    _draw_board_base(ax, board)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Influence φ')

    sym = PIECE_SYMBOLS[piece]
    col_letter = 'abcdefgh'[c]
    rank = 8 - r
    ax.set_title(f'Influence of {sym} at {col_letter}{rank}  (α={alpha})', fontsize=12)
    fig.tight_layout()
    return fig


def plot_all_fields(
    fen: str,
    alpha: float = 0.5,
    figsize: tuple = (18, 6),
) -> plt.Figure:
    """
    Side-by-side: white field | black field | control field.
    """
    fields = compute_fields_from_fen(fen, alpha=alpha)
    board  = fields['board']

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    configs = [
        ('white',   'Blues',  'White Influence Φ_w(x)'),
        ('black',   'Reds',   'Black Influence Φ_b(x)'),
        ('control', 'RdBu',   'Control C(x) = Φ_w − Φ_b'),
    ]

    for ax, (key, cmap, title) in zip(axes, configs):
        field   = fields[key]
        display = field[::-1, :]
        vmax    = max(abs(display).max(), 1e-6)
        vmin    = 0 if key != 'control' else -vmax

        im = ax.imshow(
            display, cmap=cmap,
            extent=[0, 8, 0, 8],
            vmin=vmin, vmax=vmax,
            alpha=0.55, zorder=1,
            interpolation='nearest'
        )
        _draw_board_base(ax, board)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=11)

    fig.suptitle(f'Chess Influence Fields  (α={alpha})', fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def plot_alpha_comparison(
    fen: str,
    alphas: list[float] = [0.3, 0.5, 0.7, 0.9],
    figsize: tuple = (20, 5),
) -> plt.Figure:
    """
    Compare control fields for different values of α side by side.
    Useful for understanding how the decay parameter affects the field.
    """
    fig, axes = plt.subplots(1, len(alphas), figsize=figsize)

    for ax, a in zip(axes, alphas):
        fields  = compute_fields_from_fen(fen, alpha=a)
        board   = fields['board']
        C       = fields['control'][::-1, :]
        vmax    = max(abs(C).max(), 1e-6)

        im = ax.imshow(
            C, cmap='RdBu',
            extent=[0, 8, 0, 8],
            vmin=-vmax, vmax=vmax,
            alpha=0.55, zorder=1,
            interpolation='nearest'
        )
        _draw_board_base(ax, board)
        ax.set_title(f'α = {a}', fontsize=12)

    fig.suptitle('Control Field for Different α Values', fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    FEN = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1'

    fig1 = plot_all_fields(FEN)
    fig1.savefig('fields_demo.png', dpi=150, bbox_inches='tight')
    print("Saved: fields_demo.png")

    fig2 = plot_alpha_comparison(FEN)
    fig2.savefig('alpha_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: alpha_comparison.png")

    plt.show()