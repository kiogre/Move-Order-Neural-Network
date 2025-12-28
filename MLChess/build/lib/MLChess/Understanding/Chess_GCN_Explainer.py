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

class ChessGCNExplainer:
    """Toolkit per interpretabilità di GCN per scacchi"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def visualize_move_probabilities(self, data, board: chess.Board, 
                                     top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualizza quali mosse la rete considera più probabili
        
        Returns:
            attention_from: heatmap 8x8 di quanto ogni casella è 'sorgente' di mosse
            attention_to: heatmap 8x8 di quanto ogni casella è 'destinazione' di mosse
        """
        # Aggiungi batch se manca
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        
        with torch.no_grad():
            logits, value = self.model(data)
            
            # Maschera mosse illegali
            if hasattr(data, 'legal_edge_mask'):
                logits = logits.masked_fill(data.legal_edge_mask == 0, -1e9)
            
            probs = F.softmax(logits, dim=0)
        
        # Crea heatmap
        attention_from = np.zeros((8, 8))
        attention_to = np.zeros((8, 8))
        
        src, dst = data.edge_index
        edge_info = []
        
        for i, (s, d, p) in enumerate(zip(src, dst, probs)):
            if data.legal_edge_mask[i]:  # Solo mosse legali
                s_sq, d_sq = s.item(), d.item()
                prob = p.item()
                
                from_rank, from_file = s_sq // 8, s_sq % 8
                to_rank, to_file = d_sq // 8, d_sq % 8
                
                attention_from[from_rank, from_file] += prob
                attention_to[to_rank, to_file] += prob
                
                edge_info.append((s_sq, d_sq, prob))
        
        # Ordina per probabilità
        edge_info.sort(key=lambda x: x[2], reverse=True)
        
        # Visualizza
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Scacchiera
        self._plot_board(board, axes[0])
        axes[0].set_title(f'Posizione (Value: {value.item():.3f})')
        
        # FROM heatmap
        im1 = axes[1].imshow(attention_from[::-1], cmap='hot', interpolation='nearest')
        axes[1].set_title('Da dove muovere')
        self._add_chess_labels(axes[1])
        plt.colorbar(im1, ax=axes[1])
        
        # TO heatmap
        im2 = axes[2].imshow(attention_to[::-1], cmap='hot', interpolation='nearest')
        axes[2].set_title('Dove muovere')
        self._add_chess_labels(axes[2])
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        
        # Stampa top-k mosse
        print(f"\nTop {top_k} mosse più probabili:")
        for i, (src, dst, prob) in enumerate(edge_info[:top_k]):
            move_uci = chess.square_name(src) + chess.square_name(dst)
            print(f"{i+1}. {move_uci}: {prob*100:.2f}%")
        
        return attention_from, attention_to, fig
    
    def gradcam_nodes(self, data, board: chess.Board, 
                     target: str = 'value') -> np.ndarray:
        """
        GradCAM sui nodi per vedere quali caselle influenzano la decisione
        
        Args:
            target: 'value' per valutazione posizione, 'move' per best move
        """
        # Aggiungi batch se manca
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        
        # Forward con gradienti
        data.x.requires_grad = True
        logits, value = self.model(data)
        
        # Scegli target
        if target == 'value':
            output = value.squeeze()
        else:  # best move
            if hasattr(data, 'legal_edge_mask'):
                logits = logits.masked_fill(data.legal_edge_mask == 0, -1e9)
            output = logits.max()
        
        # Backward
        self.model.zero_grad()
        output.backward()
        
        # Gradiente rispetto ai nodi
        gradients = data.x.grad  # [64, 15]
        
        # Importanza: somma valori assoluti dei gradienti
        node_importance = gradients.abs().sum(dim=1)  # [64]
        node_importance = node_importance / node_importance.max()
        
        # Reshape in scacchiera
        heatmap = node_importance.detach().cpu().numpy().reshape(8, 8)
        
        # Visualizza
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        self._plot_board(board, axes[0])
        axes[0].set_title('Posizione')
        
        im = axes[1].imshow(heatmap[::-1], cmap='hot', interpolation='nearest')
        axes[1].set_title(f'GradCAM ({target})')
        self._add_chess_labels(axes[1])
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        
        return heatmap, fig
    
    def integrated_gradients(self, data, board: chess.Board, 
                            steps: int = 50) -> np.ndarray:
        """
        Integrated Gradients per attributions più robuste
        """
        # Aggiungi batch se manca
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        
        # Baseline: scacchiera vuota
        baseline = torch.zeros_like(data.x)
        
        alphas = torch.linspace(0, 1, steps)
        gradients = []
        
        for alpha in alphas:
            # Interpolazione
            interpolated = baseline + alpha * (data.x - baseline)
            interpolated = interpolated.detach().clone()
            interpolated.requires_grad = True
            
            # Crea nuovo data object
            data_interp = data.clone()
            data_interp.x = interpolated
            
            # Forward
            logits, value = self.model(data_interp)
            output = value.squeeze()
            
            # Backward
            self.model.zero_grad()
            output.backward()
            
            gradients.append(interpolated.grad.detach())
        
        # Media gradienti
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Integrated gradients
        integrated_grads = (data.x - baseline) * avg_gradients
        
        # Importanza per nodo
        node_importance = integrated_grads.abs().sum(dim=1)
        node_importance = node_importance / node_importance.max()
        heatmap = node_importance.cpu().numpy().reshape(8, 8)
        
        # Visualizza
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        self._plot_board(board, axes[0])
        axes[0].set_title('Posizione')
        
        im = axes[1].imshow(heatmap[::-1], cmap='hot', interpolation='nearest')
        axes[1].set_title('Integrated Gradients')
        self._add_chess_labels(axes[1])
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        
        return heatmap, fig
    
    def analyze_embeddings(self, data, board: chess.Board) -> np.ndarray:
        """
        Analizza l'attivazione dei nodi dopo le GCN layers
        """
        # Aggiungi batch se manca
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        
        with torch.no_grad():
            # Estrai embedding intermedie
            x = F.relu(self.model.GCN.input_proj(data.x))
            x = self.model.GCN.conv1(x, data.edge_index)
            x = F.relu(x)
            x = self.model.GCN.conv2(x, data.edge_index)
            node_embeddings = F.relu(x)
        
        # Norma delle embedding (quanto è "attivo" ogni nodo)
        activation = node_embeddings.norm(dim=1)
        activation = activation / activation.max()
        heatmap = activation.cpu().numpy().reshape(8, 8)
        
        # Visualizza
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        self._plot_board(board, axes[0])
        axes[0].set_title('Posizione')
        
        im = axes[1].imshow(heatmap[::-1], cmap='viridis', interpolation='nearest')
        axes[1].set_title('Attivazione nodi (dopo GCN)')
        self._add_chess_labels(axes[1])
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        
        return heatmap, fig
    
    def complete_analysis(self, data, board: chess.Board, save_path: Optional[str] = None):
        """
        Analisi completa: tutte le visualizzazioni in una figura
        """
        # Aggiungi batch se manca
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Scacchiera originale
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_board(board, ax1)
        with torch.no_grad():
            _, value = self.model(data)
        ax1.set_title(f'Posizione\nValue: {value.item():.3f}', fontsize=10)
        
        # 2-3. Move probabilities
        with torch.no_grad():
            logits, _ = self.model(data)
            if hasattr(data, 'legal_edge_mask'):
                logits = logits.masked_fill(data.legal_edge_mask == 0, -1e9)
            probs = F.softmax(logits, dim=0)
        
        attention_from = np.zeros((8, 8))
        attention_to = np.zeros((8, 8))
        src, dst = data.edge_index
        
        for i, (s, d, p) in enumerate(zip(src, dst, probs)):
            if hasattr(data, 'legal_edge_mask') and data.legal_edge_mask[i]:
                s_sq, d_sq = s.item(), d.item()
                attention_from[s_sq // 8, s_sq % 8] += p.item()
                attention_to[d_sq // 8, d_sq % 8] += p.item()
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(attention_from[::-1], cmap='hot', interpolation='nearest')
        ax2.set_title('Move Attention FROM', fontsize=10)
        self._add_chess_labels(ax2)
        plt.colorbar(im2, ax=ax2)
        
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(attention_to[::-1], cmap='hot', interpolation='nearest')
        ax3.set_title('Move Attention TO', fontsize=10)
        self._add_chess_labels(ax3)
        plt.colorbar(im3, ax=ax3)
        
        # 4. GradCAM (value)
        data.x.requires_grad = True
        logits, value = self.model(data)
        self.model.zero_grad()
        value.backward()
        gradients = data.x.grad
        node_importance = gradients.abs().sum(dim=1)
        heatmap_value = (node_importance / node_importance.max()).detach().cpu().numpy().reshape(8, 8)
        
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(heatmap_value[::-1], cmap='hot', interpolation='nearest')
        ax4.set_title('GradCAM (Value)', fontsize=10)
        self._add_chess_labels(ax4)
        plt.colorbar(im4, ax=ax4)
        
        # 5. GradCAM (move)
        data.x.requires_grad = True
        logits, _ = self.model(data)
        if hasattr(data, 'legal_edge_mask'):
            logits = logits.masked_fill(data.legal_edge_mask == 0, -1e9)
        self.model.zero_grad()
        logits.max().backward()
        gradients = data.x.grad
        node_importance = gradients.abs().sum(dim=1)
        heatmap_move = (node_importance / node_importance.max()).detach().cpu().numpy().reshape(8, 8)
        
        ax5 = fig.add_subplot(gs[1, 0])
        im5 = ax5.imshow(heatmap_move[::-1], cmap='hot', interpolation='nearest')
        ax5.set_title('GradCAM (Best Move)', fontsize=10)
        self._add_chess_labels(ax5)
        plt.colorbar(im5, ax=ax5)
        
        # 6. Integrated Gradients
        baseline = torch.zeros_like(data.x)
        alphas = torch.linspace(0, 1, 30)
        gradients_list = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (data.x - baseline)
            interpolated = interpolated.detach().clone()
            interpolated.requires_grad = True
            data_interp = data.clone()
            data_interp.x = interpolated
            _, value = self.model(data_interp)
            self.model.zero_grad()
            value.backward()
            gradients_list.append(interpolated.grad.detach())
        
        avg_grads = torch.stack(gradients_list).mean(dim=0)
        ig = (data.x - baseline) * avg_grads
        ig_importance = ig.abs().sum(dim=1) / ig.abs().sum(dim=1).max()
        heatmap_ig = ig_importance.detach().cpu().numpy().reshape(8, 8)
        
        ax6 = fig.add_subplot(gs[1, 1])
        im6 = ax6.imshow(heatmap_ig[::-1], cmap='hot', interpolation='nearest')
        ax6.set_title('Integrated Gradients', fontsize=10)
        self._add_chess_labels(ax6)
        plt.colorbar(im6, ax=ax6)
        
        # 7. Node embeddings activation
        with torch.no_grad():
            x = F.relu(self.model.GCN.input_proj(data.x))
            x = self.model.GCN.conv1(x, data.edge_index)
            x = F.relu(x)
            x = self.model.GCN.conv2(x, data.edge_index)
            embeddings = F.relu(x)
        
        activation = embeddings.norm(dim=1) / embeddings.norm(dim=1).max()
        heatmap_emb = activation.cpu().numpy().reshape(8, 8)
        
        ax7 = fig.add_subplot(gs[1, 2])
        im7 = ax7.imshow(heatmap_emb[::-1], cmap='viridis', interpolation='nearest')
        ax7.set_title('Node Activation (GCN)', fontsize=10)
        self._add_chess_labels(ax7)
        plt.colorbar(im7, ax=ax7)
        
        # 8. Feature importance per tipo
        feature_names = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k', 'file', 'rank', 'value']
        feature_importance = gradients.abs().mean(dim=0).cpu().numpy()
        
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.barh(range(len(feature_names)), feature_importance)
        ax8.set_yticks(range(len(feature_names)))
        ax8.set_yticklabels(feature_names, fontsize=8)
        ax8.set_xlabel('Importanza media', fontsize=8)
        ax8.set_title('Feature Importance', fontsize=10)
        
        # 9. Top moves table
        ax9 = fig.add_subplot(gs[2, :])
        ax9.axis('off')
        
        with torch.no_grad():
            logits, _ = self.model(data)
            if hasattr(data, 'legal_edge_mask'):
                logits = logits.masked_fill(data.legal_edge_mask == 0, -1e9)
            probs = F.softmax(logits, dim=0)
        
        edge_info = []
        src, dst = data.edge_index
        for i, (s, d, p) in enumerate(zip(src, dst, probs)):
            if hasattr(data, 'legal_edge_mask') and data.legal_edge_mask[i]:
                edge_info.append((s.item(), d.item(), p.item()))
        
        edge_info.sort(key=lambda x: x[2], reverse=True)
        
        table_data = []
        for i, (s, d, prob) in enumerate(edge_info[:10]):
            move = chess.square_name(s) + chess.square_name(d)
            table_data.append([f"{i+1}", move, f"{prob*100:.2f}%"])
        
        table = ax9.table(cellText=table_data, 
                         colLabels=['Rank', 'Move', 'Probability'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.1, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax9.set_title('Top 10 Predicted Moves', fontsize=12, pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Salvato in {save_path}")
        
        return fig
    
    def _plot_board(self, board: chess.Board, ax):
        """Disegna scacchiera usando chess.svg per rendering di qualità"""
        try:
            # Genera SVG della scacchiera
            svg_data = chess.svg.board(board, size=400)
            
            # Converti SVG in PNG usando cairosvg
            png_data = svg2png(bytestring=svg_data.encode('utf-8'))
            
            # Carica come immagine PIL
            img = Image.open(BytesIO(png_data))
            
            # Mostra in matplotlib
            ax.imshow(img)
            ax.axis('off')
            
        except ImportError:
            # Fallback al metodo vecchio se cairosvg non è installato
            print("Tip: installa cairosvg per rendering migliore: pip install cairosvg")
            self._plot_board_simple(board, ax)
    
    def _plot_board_simple(self, board: chess.Board, ax):
        """Fallback: disegna scacchiera con matplotlib (meno bella)"""
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        
        # Colori caselle
        for rank in range(8):
            for file in range(8):
                color = '#F0D9B5' if (rank + file) % 2 == 0 else '#B58863'
                rect = Rectangle((file, 7-rank), 1, 1, facecolor=color)
                ax.add_patch(rect)
        
        # Pezzi (Unicode)
        piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = square // 8, square % 8
                symbol = piece_symbols[piece.symbol()]
                ax.text(file + 0.5, 7 - rank + 0.5, symbol, 
                       ha='center', va='center', fontsize=24)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _add_chess_labels(self, ax):
        """Aggiunge coordinate scacchi"""
        ax.set_xticks(np.arange(8) + 0.5)
        ax.set_yticks(np.arange(8) + 0.5)
        ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])


# ===== ESEMPIO DI UTILIZZO =====
"""
# 1. Carica modello e dati
model = TestModelGCN(hidden_dim=256)
model.load_state_dict(torch.load('your_model.pth'))

# 2. Crea explainer
explainer = ChessGCNExplainer(model)

# 3. Analizza una posizione
board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
converter = ChessPositionGraph()
data = converter.fen_to_graph(board.fen(), "+50", None)

# 4. Analisi completa
fig = explainer.complete_analysis(data, board, save_path='analysis.png')
plt.show()

# Oppure analisi specifiche:
# explainer.visualize_move_probabilities(data, board)
# explainer.gradcam_nodes(data, board, target='value')
# explainer.integrated_gradients(data, board)
"""