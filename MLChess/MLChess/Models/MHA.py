import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, d_model = 64*7, n_heads = 7):
        super(MHA, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        matrices_moves = torch.load('attention_based_matrix_64x64.pt')
        self.register_buffer('matrices_moves', matrices_moves)
        self.lambda_ = nn.Parameter(torch.ones(7))

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear projections
        Q = self.q_linear(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        scores = scores + self.lambda_.view(1, 7, 1, 1) * self.matrices_moves

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(attn_output)
        
        return output
    

class ChessMHA(nn.Module):
    '''
    Per iniziare lo faccio piccolino, se funziona forse aumenterò i layer in mezzo
    '''
    def __init__(self, d_model=64*7, n_heads=7):
        super(ChessMHA, self).__init__()
        self.mha = MHA(d_model, n_heads)
        self.layer_1 = nn.Linear(d_model, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.value = nn.Linear(128, 1)
        self.policy = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, 1968)
        )

    def forward(self, x):
        x = self.mha(x)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)

        value = self.value(x)
        policy = self.policy(x)

        return value, policy