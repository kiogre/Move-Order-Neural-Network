import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, d_model = 64*7, n_heads = 7, input_dim = 13):
        super(MHA, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        matrices_moves = torch.load('attention_based_matrix_64x64.pt')
        self.register_buffer('matrices_moves', matrices_moves)
        self.lambda_ = nn.Parameter(torch.ones(7))

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        self.q_linear = nn.Linear(input_dim, d_model)
        self.k_linear = nn.Linear(input_dim, d_model)
        self.v_linear = nn.Linear(input_dim, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear projections
        Q = self.q_linear(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        #scores = scores + self.lambda_.view(1, 7, 1, 1) * self.matrices_moves
        scores = scores + self.matrices_moves

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(attn_output)
        
        return output
    


class MHA_2(nn.Module):
    """
    Multi-Head Attention con prior cinematico moltiplicativo, parallelizzato.
 
    Formula:
        x_kin_i  = M_i @ x                        (batch, 7, 64, input_dim)
        Q_i      = x_kin_i @ W_q^(i)              (batch, 7, 64, head_dim)
        K_i      = x_kin_i @ W_k^(i)              (batch, 7, 64, head_dim)
        V_i      = x        @ W_v  → reshape       (batch, 7, 64, head_dim)
 
        scores_i = Q_i @ K_i^T / sqrt(d_head)
        out      = concat(softmax(scores_i) @ V_i) @ W_o
    """
 
    def __init__(self, d_model: int = 64 * 7, n_heads: int = 7, input_dim: int = 13):
        super().__init__()
        assert d_model % n_heads == 0, "d_model deve essere divisibile per n_heads"
 
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads  # 64
 
        # Matrici cinematiche — (7, 64, 64), non learnable
        M = torch.load('attention_based_matrix_64x64.pt')
        self.register_buffer('M', M)
 
        # Pesi Q e K — shape (7, input_dim, head_dim), una per testa
        self.W_q = nn.Parameter(torch.randn(n_heads, input_dim, self.head_dim) * 0.02)
        self.W_k = nn.Parameter(torch.randn(n_heads, input_dim, self.head_dim) * 0.02)
 
        # V — proiezione standard condivisa, non filtrata cinematicamente
        self.W_v = nn.Linear(input_dim, d_model, bias=False)
 
        # Proiezione di output
        self.W_o = nn.Linear(d_model, d_model)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 64, input_dim)
 
        Returns:
            out: (batch, 64, d_model)
        """
        batch = x.size(0)
 
        # Applica tutti gli M_i contemporaneamente
        # M: (7, 64, 64), x: (batch, 64, input_dim)
        # → (batch, 7, 64, input_dim)
        x_kin = torch.einsum('hsc,bcd->bhsd', self.M, x)
 
        # Proiezioni Q e K per ogni testa in parallelo
        # W_q: (7, input_dim, head_dim)
        # → (batch, 7, 64, head_dim)
        Q = torch.einsum('bhsd,hde->bhse', x_kin, self.W_q)
        K = torch.einsum('bhsd,hde->bhse', x_kin, self.W_k)
 
        # V — proiezione standard poi reshape
        # (batch, 64, d_model) → (batch, 7, 64, head_dim)
        V = self.W_v(x).view(batch, 64, self.n_heads, self.head_dim).transpose(1, 2)
 
        # Scaled dot-product attention
        # → (batch, 7, 64, 64)
        scores = torch.einsum('bhsd,bhtd->bhst', Q, K) / (self.head_dim ** 0.5)
        attn   = F.softmax(scores, dim=-1)
 
        # Applica attenzione a V
        # → (batch, 7, 64, head_dim)
        out = torch.einsum('bhst,bhtd->bhsd', attn, V)
 
        # Concatena teste e proietta
        # (batch, 7, 64, head_dim) → (batch, 64, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, 64, self.d_model)
        return self.W_o(out)
    
class MHA_3(nn.Module):
    """
    Multi-Head Attention piece-centric.
 
    Per ogni testa i (tipo di pezzo):
        1. Calcola attenzione completa (64x64) come normale
        2. Seleziona solo le righe corrispondenti alle caselle
           occupate da pezzi di tipo i
        3. Mean pooling su quelle righe → vettore (64, head_dim)
 
    Questo significa: "cosa vedono in media i pezzi di tipo i?"
    invece di "cosa vede ogni casella?"
 
    Le 7 teste corrispondono all'ordine delle matrici:
        0: pedone bianco, 1: pedone nero, 2: cavallo,
        3: alfiere, 4: torre, 5: regina, 6: re
    """
 
    HEAD_TO_PLANES = {
        0: [0],       # pedone bianco
        1: [6],       # pedone nero
        2: [1, 7],    # cavallo bianco + nero
        3: [2, 8],    # alfiere bianco + nero
        4: [3, 9],    # torre bianca + nera
        5: [4, 10],   # regina bianca + nera
        6: [5, 11],   # re bianco + nero
    }
 
    def __init__(self, d_model: int = 64 * 7, n_heads: int = 7, input_dim: int = 13):
        super().__init__()
        assert d_model % n_heads == 0
 
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.head_dim  = d_model // n_heads  # 64
        self.input_dim = input_dim
 
        matrices_moves = torch.load('attention_based_matrix_64x64.pt')
        self.register_buffer('matrices_moves', matrices_moves)
        self.lambda_ = nn.Parameter(torch.ones(7))
 
        self.q_linear   = nn.Linear(input_dim, d_model)
        self.k_linear   = nn.Linear(input_dim, d_model)
        self.v_linear   = nn.Linear(input_dim, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 64, input_dim)
 
        Returns:
            output: (batch, 64, d_model)
        """
        batch = x.size(0)
 
        # Proiezioni complete su tutte le 64 caselle
        Q = self.q_linear(x).view(batch, 64, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch, 64, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch, 64, self.n_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: (batch, n_heads, 64, head_dim)
 
        # Scores e attenzione completi — (batch, n_heads, 64, 64)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
 
        # Bias cinematico opzionale — decommenta per attivarlo
        # scores = scores + self.lambda_.view(1, 7, 1, 1) * self.matrices_moves
 
        attn = F.softmax(scores, dim=-1)
 
        # Output completo — (batch, n_heads, 64, head_dim)
        full_out = torch.matmul(attn, V)
 
        head_outputs = []
 
        for i in range(self.n_heads):
            # Maschera dei pezzi di tipo i — (batch, 64)
            planes = self.HEAD_TO_PLANES[i]
            piece_mask = x[:, :, planes[0]].clone()
            for p in planes[1:]:
                piece_mask = piece_mask + x[:, :, p]
            piece_mask = piece_mask.clamp(max=1.0)
 
            # full_out[:, i]: (batch, 64, head_dim)
            # Azzera le righe delle caselle senza il pezzo
            out_i = full_out[:, i] * piece_mask.unsqueeze(-1)  # (batch, 64, head_dim)
 
            # Mean pooling solo sulle caselle con il pezzo
            n_pieces = piece_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1)
            pooled = out_i.sum(dim=1) / n_pieces  # (batch, head_dim)
 
            # Espandi a (batch, 64, head_dim) per concatenazione finale
            #pooled_expanded = pooled.unsqueeze(1).expand(-1, 64, -1)  # (batch, 64, head_dim)
            #head_outputs.append(pooled_expanded)
            head_outputs.append(pooled)
 
        # Concatena le 7 teste → (batch, 64, d_model)
        attn_output = torch.cat(head_outputs, dim=-1)
        return self.out_linear(attn_output)

class ChessMHA(nn.Module):
    '''
    Per iniziare lo faccio piccolino, se funziona forse aumenterò i layer in mezzo
    '''
    def __init__(self, d_model=64*7, n_heads=7):
        super(ChessMHA, self).__init__()
        self.mha = MHA(d_model, n_heads)
        self.layer_1 = nn.Linear(d_model*64, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.value = nn.Linear(128, 1)
        self.policy = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, 1968)
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, mask):

        batch_size = x.size(0)
        x = x.view(batch_size, 13, 64).permute(0, 2, 1)  # (batch, 64, 13)

        x = self.mha(x)

        x = x.view(batch_size, -1)

        x = self.layer_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.dropout(x)

        value = self.value(x)
        policy = self.policy(x)

        mask = mask.bool()
        policy = policy.masked_fill(mask == 0, float('-inf'))

        return value, policy
    

class ChessMHA_2(nn.Module):
    '''
    Per iniziare lo faccio piccolino, se funziona forse aumenterò i layer in mezzo
    '''
    def __init__(self, d_model=64*7, n_heads=7):
        super(ChessMHA_2, self).__init__()
        self.mha = MHA_2(d_model, n_heads)
        self.layer_1 = nn.Linear(d_model*64, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.value = nn.Linear(128, 1)
        self.policy = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, 1968)
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, mask):

        batch_size = x.size(0)
        x = x.view(batch_size, 13, 64).permute(0, 2, 1)  # (batch, 64, 13)

        x = self.mha(x)

        x = x.view(batch_size, -1)

        x = self.layer_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.dropout(x)

        value = self.value(x)
        policy = self.policy(x)

        mask = mask.bool()
        policy = policy.masked_fill(mask == 0, float('-inf'))

        return value, policy


class ChessMHA_3(nn.Module):
    '''
    Per iniziare lo faccio piccolino, se funziona forse aumenterò i layer in mezzo
    '''
    def __init__(self, d_model=64*7, n_heads=7):
        super(ChessMHA_3, self).__init__()
        self.mha = MHA_3(d_model, n_heads)
        self.layer_1 = nn.Linear(d_model, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.value = nn.Linear(128, 1)
        self.policy = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, 1968)
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, mask):

        batch_size = x.size(0)
        x = x.view(batch_size, 13, 64).permute(0, 2, 1)  # (batch, 64, 13)

        x = self.mha(x)

        #x = x.view(batch_size, -1)

        x = self.layer_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.dropout(x)

        value = self.value(x)
        policy = self.policy(x)

        mask = mask.bool()
        policy = policy.masked_fill(mask == 0, float('-inf'))

        return value, policy