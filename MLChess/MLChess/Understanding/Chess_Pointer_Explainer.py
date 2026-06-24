import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import chess
import chess.svg
from matplotlib.patches import Rectangle
from typing import Tuple, Optional
from io import BytesIO
from cairosvg import svg2png
from PIL import Image

from ..Models.Pointer_model import JellyFishPointer  # adatta il percorso se necessario


class ChessPointerExplainer:
    """Toolkit per interpretabilità del Pointer Network per scacchi"""
    
    def __init__(self, model: JellyFishPointer):
        self.model = model
        self.model.eval()
    
    def visualize_move_probabilities(self, 
                                     board_tensor: torch.Tensor,
                                     legal_moves: torch.Tensor,
                                     move_mask: torch.Tensor | None = None,
                                     board: chess.Board = None,
                                     top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualizza le probabilità assegnate dal Pointer Network.
        """
        if board_tensor.dim() == 3:
            board_tensor = board_tensor.unsqueeze(0)  # aggiungi batch
        if legal_moves.dim() == 2:
            legal_moves = legal_moves.unsqueeze(0)
        if move_mask is not None and move_mask.dim() == 1:
            move_mask = move_mask.unsqueeze(0)

        with torch.no_grad():
            logits, probs, value = self.model(board_tensor, legal_moves, move_mask)
            probs = probs.squeeze(0)  # (N_moves,)

        # Heatmap From / To
        attention_from = np.zeros((8, 8))
        attention_to = np.zeros((8, 8))

        moves_list = list(board.legal_moves) if board else []

        move_info = []
        for i, prob in enumerate(probs):
            if i >= len(moves_list):
                break
            move = moves_list[i]
            prob_val = prob.item()

            from_sq = move.from_square
            to_sq = move.to_square

            from_rank, from_file = divmod(from_sq, 8)
            to_rank, to_file = divmod(to_sq, 8)

            attention_from[from_rank, from_file] += prob_val
            attention_to[to_rank, to_file] += prob_val

            move_info.append((from_sq, to_sq, prob_val, chess.square_name(from_sq) + chess.square_name(to_sq)))

        # Ordina per probabilità
        move_info.sort(key=lambda x: x[2], reverse=True)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        if board:
            self._plot_board(board, axes[0])
            axes[0].set_title(f'Posizione (Value: {value.item():.3f})')
        else:
            axes[0].set_title('Posizione')

        im1 = axes[1].imshow(attention_from[::-1], cmap='hot', interpolation='nearest')
        axes[1].set_title('Da dove muovere (From)')
        self._add_chess_labels(axes[1])
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(attention_to[::-1], cmap='hot', interpolation='nearest')
        axes[2].set_title('Dove muovere (To)')
        self._add_chess_labels(axes[2])
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()

        print(f"\nTop {top_k} mosse più probabili:")
        for i, (_, _, prob, uci) in enumerate(move_info[:top_k]):
            print(f"{i+1:2d}. {uci}: {prob*100:5.2f}%")

        return attention_from, attention_to, fig

    def gradcam_backbone(self, 
                        board_tensor: torch.Tensor,
                        legal_moves: torch.Tensor,
                        move_mask: torch.Tensor | None = None,
                        board: chess.Board = None,
                        target: str = 'value') -> np.ndarray:
        """
        GradCAM sul backbone CNN.
        """
        if board_tensor.dim() == 3:
            board_tensor = board_tensor.unsqueeze(0)
        if legal_moves.dim() == 2:
            legal_moves = legal_moves.unsqueeze(0)
        if move_mask is not None and move_mask.dim() == 1:
            move_mask = move_mask.unsqueeze(0)

        board_tensor = board_tensor.requires_grad_(True)

        logits, probs, value = self.model(board_tensor, legal_moves, move_mask)

        if target == 'value':
            output = value.squeeze()
        else:  # policy
            output = logits.max()

        self.model.zero_grad()
        output.backward()

        # Gradienti rispetto all'input board (13 canali)
        gradients = board_tensor.grad[0]           # (13, 8, 8)
        importance = gradients.abs().sum(dim=0)    # (8, 8)
        importance = importance / (importance.max() + 1e-8)

        heatmap = importance.detach().cpu().numpy()

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        if board:
            self._plot_board(board, axes[0])
            axes[0].set_title('Posizione')
        
        im = axes[1].imshow(heatmap[::-1], cmap='hot', interpolation='nearest')
        axes[1].set_title(f'GradCAM - {target.capitalize()}')
        self._add_chess_labels(axes[1])
        plt.colorbar(im, ax=axes[1])
        plt.tight_layout()

        return heatmap, fig

    def feature_maps(self, board_tensor: torch.Tensor, board: chess.Board = None, layer: str = 'layer4'):
        """Visualizza le feature map del backbone (utile per debugging)"""
        # Questo richiede un hook. Per semplicità mostriamo attivazione dopo conv1 e dopo layer4
        activations = {}

        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook

        # Registra hook temporanei
        hooks = []
        hooks.append(self.model.backbone.conv1.register_forward_hook(hook_fn('conv1')))
        hooks.append(self.model.backbone.layer4.register_forward_hook(hook_fn('layer4')))

        with torch.no_grad():
            self.model.backbone(board_tensor.unsqueeze(0))

        for h in hooks:
            h.remove()

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        if board:
            self._plot_board(board, axes[0])

        # Media delle feature map di layer4
        fmap = activations['layer4'][0].mean(dim=0).cpu().numpy()  # media sui canali
        im = axes[1].imshow(fmap[::-1], cmap='viridis')
        axes[1].set_title('Media Feature Map Layer4')
        self._add_chess_labels(axes[1])
        plt.colorbar(im, ax=axes[1])

        plt.tight_layout()
        return fig

    def complete_analysis(self, 
                         board_tensor: torch.Tensor,
                         legal_moves: torch.Tensor,
                         move_mask: torch.Tensor | None = None,
                         board: chess.Board = None,
                         save_path: Optional[str] = None):
        """Analisi completa in una sola figura"""
        fig = plt.figure(figsize=(22, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. Scacchiera
        ax1 = fig.add_subplot(gs[0, 0])
        if board:
            self._plot_board(board, ax1)
            with torch.no_grad():
                _, _, value = self.model(board_tensor.unsqueeze(0), 
                                       legal_moves.unsqueeze(0) if legal_moves.dim()==2 else legal_moves,
                                       move_mask.unsqueeze(0) if move_mask is not None else None)
            ax1.set_title(f'Posizione\nValue: {value.item():.3f}')
        else:
            ax1.set_title('Posizione')

        # 2-3. Move Probabilities
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        
        att_from, att_to, _ = self.visualize_move_probabilities(
            board_tensor, legal_moves, move_mask, board=None
        )
        ax2.imshow(att_from[::-1], cmap='hot')
        ax2.set_title('Attention FROM')
        self._add_chess_labels(ax2)
        
        ax3.imshow(att_to[::-1], cmap='hot')
        ax3.set_title('Attention TO')
        self._add_chess_labels(ax3)

        # 4. GradCAM Value
        _, fig_gc_value = self.gradcam_backbone(board_tensor, legal_moves, move_mask, board=None, target='value')
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(fig_gc_value.axes[1].get_images()[0].get_array(), cmap='hot')
        ax4.set_title('GradCAM (Value)')
        self._add_chess_labels(ax4)

        # 5. GradCAM Policy
        _, fig_gc_policy = self.gradcam_backbone(board_tensor, legal_moves, move_mask, board=None, target='policy')
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(fig_gc_policy.axes[1].get_images()[0].get_array(), cmap='hot')
        ax5.set_title('GradCAM (Best Move)')
        self._add_chess_labels(ax5)

        # 6. Feature Map
        ax6 = fig.add_subplot(gs[1, 1])
        # Placeholder per feature map (puoi raffinare)
        ax6.set_title('Feature Maps (Layer 4)')
        ax6.axis('off')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"✅ Analisi salvata in: {save_path}")
        
        return fig

    # ====================== Utility ======================
    def _plot_board(self, board: chess.Board, ax):
        try:
            svg_data = chess.svg.board(board, size=400)
            png_data = svg2png(bytestring=svg_data.encode('utf-8'))
            img = Image.open(BytesIO(png_data))
            ax.imshow(img)
            ax.axis('off')
        except ImportError:
            self._plot_board_simple(board, ax)

    def _plot_board_simple(self, board: chess.Board, ax):
        # fallback semplice (stesso del GCN explainer)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        for rank in range(8):
            for file in range(8):
                color = '#F0D9B5' if (rank + file) % 2 == 0 else '#B58863'
                rect = Rectangle((file, 7-rank), 1, 1, facecolor=color)
                ax.add_patch(rect)
        
        piece_symbols = {'P':'♙','N':'♘','B':'♗','R':'♖','Q':'♕','K':'♔',
                        'p':'♟','n':'♞','b':'♝','r':'♜','q':'♛','k':'♚'}
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                ax.text(file + 0.5, 7 - rank + 0.5, 
                       piece_symbols[piece.symbol()], 
                       ha='center', va='center', fontsize=28)

    def _add_chess_labels(self, ax):
        ax.set_xticks(np.arange(8) + 0.5)
        ax.set_yticks(np.arange(8) + 0.5)
        ax.set_xticklabels(['a','b','c','d','e','f','g','h'])
        ax.set_yticklabels(['8','7','6','5','4','3','2','1'])