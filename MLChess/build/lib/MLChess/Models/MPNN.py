from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessMPNN(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='add')  # sum aggregation

        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: features del nodo sorgente
        # edge_attr: tipo di mossa (queen, knight, distanza, direzione, ecc.)
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.msg_mlp(msg_input)

    def update(self, aggr_out, x):
        # aggr_out: messaggi aggregati
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))
