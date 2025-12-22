from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
import torch.nn as nn
import torch.nn.functional as F
import torch

class ChessGCN(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=256, global_dim = 7):  # Più largo
        super().__init__()
        self.global_dim = global_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        self.global_fc = nn.Linear(global_dim, hidden_dim // 2)

        combined_dim = hidden_dim*3 + hidden_dim // 2
        # QUESTO è dove spendi GPU - dense layers enormi
        self.dense = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 4),  # 3 pooling types
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x, edge_index, batch, global_features):
        if batch.numel() == 0:
            batch_size = 1
        else:
            batch_size = batch.max().item() + 1
            
        if global_features.dim() == 1:
            # Se è un vettore lungo, reshapa
            global_features = global_features.view(batch_size, self.global_dim)
        # Node features
        x = F.relu(self.input_proj(x))
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Graph-level pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        node_repr = torch.cat([x_mean, x_max, x_add], dim=1)
        
        # Global features processing
        global_repr = F.relu(self.global_fc(global_features))
        
        # Combina node + global representations
        combined = torch.cat([node_repr, global_repr], dim=1)

        
        # Output
        x = F.relu(self.dense(combined))
        
        return x
    

class IncompleteChessGCN(nn.Module):
    """
    This class is just some GCN layer for be used again just in case to use again, I can just call this
    class, the return is the tensor of informations (combined after all the pools...)
    REMEMBER, self.combined_dim = hidden_dim*3 + hidden_dim //2, TO KNOW HOW BIG IT IS THE COMBINED INFORMATIONS
    """
    def __init__(self, input_dim=13, hidden_dim=256, global_dim = 7):  # Più largo
        super().__init__()
        self.global_dim = global_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        self.global_fc = nn.Linear(global_dim, hidden_dim // 2)

        self.combined_dim = hidden_dim*3 + hidden_dim // 2
        
    def forward(self, x, edge_index, batch, global_features):
        if batch.numel() == 0:
            batch_size = 1
        else:
            batch_size = batch.max().item() + 1
            
        if global_features.dim() == 1:
            # Se è un vettore lungo, reshapa
            global_features = global_features.view(batch_size, self.global_dim)
        # Node features
        x = F.relu(self.input_proj(x))
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Graph-level pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        node_repr = torch.cat([x_mean, x_max, x_add], dim=1)
        
        # Global features processing
        global_repr = F.relu(self.global_fc(global_features))
        
        # Combina node + global representations
        combined = torch.cat([node_repr, global_repr], dim=1)
        
        return combined
    
