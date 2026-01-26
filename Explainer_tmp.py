from MLChess import GraphAndPoolingChessMPNN, ChessMPNNExplainer, ChessPositionGraph, ChessPositionGraphMPNN
import torch
import torch.nn as nn


class TestModelMPNN(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.MPNN = GraphAndPoolingChessMPNN(hidden_dim=self.hidden_dim)

        self.choose_arch = nn.Sequential(
            nn.Linear(3*self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(self.hidden_dim, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim*2 + hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    
    def forward(self, data):

        graph, combined = self.MPNN(data.x, data.edge_index, data.edge_attr, data.batch, data.global_features)

        src, dst = data.edge_index
        edge_emb = torch.cat([graph[src], graph[dst], graph[dst]-graph[src]], dim=1)

        logits = self.choose_arch(edge_emb).squeeze(-1)
        if hasattr(data, "legal_edge_mask"):
            logits = logits.masked_fill(
                data.legal_edge_mask == 0, -1e9
            )

        value = self.value_head(combined)

        return logits, value

# Carica il tuo modello
model = TestModelMPNN(hidden_dim=256)

checkpoint_load = torch.load('./MPNN/epoch_100.pt', weights_only=False)
model.load_state_dict(checkpoint_load["model_state"])

# Crea explainer
explainer = ChessMPNNExplainer(model)

import chess
import matplotlib.pyplot as plt

# Posizione interessante
#Best move is to take a piece: 3r2k1/p5pp/5p2/2p1n2P/2q1NP2/Pp1P2Q1/1P4P1/1KR5 b - - 1 39, -200, c4d3
#Normal test I use: 1kr5/pp2p3/2p1Q3/8/7q/2B1N3/PP6/2K5 w - - 5, #+2, c3e5
board = chess.Board("1kr5/pp2p3/2p1Q3/8/7q/2B1N3/PP6/2K5 w - - 5")

# Converti in grafo
converter = ChessPositionGraphMPNN()
data = converter.fen_to_graph(board.fen(), "#+2", "c3e5")  # esempio

# ANALISI COMPLETA (tutto in una figura)
fig = explainer.complete_analysis(data, board, save_path='position_analysis_MPNN.png')
plt.show()