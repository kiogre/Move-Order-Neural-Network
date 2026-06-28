import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from .graph_models import PoolingChessGCN

from .my_resnet import ChessBackbone

MOVE_VEC_DIM = 46
 
class MoveEncoder(nn.Module):
    """
    MLP che proietta un vettore mossa (46-dim) in uno spazio denso.
 
    Input:  [*, 46]
    Output: [*, move_emb_dim]
    """
    def __init__(self, move_emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MOVE_VEC_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, move_emb_dim),
            nn.ReLU(),
        )
 
    def forward(self, moves: torch.Tensor) -> torch.Tensor:
        # moves: [batch, n_mosse, 46]  oppure  [batch, 46]
        return self.net(moves)

class PointerPolicyHead(nn.Module):
    """
    Dato il contesto h dal backbone e gli embedding delle mosse legali,
    restituisce una distribuzione di probabilità sulle mosse legali.
 
    Args:
        backbone_dim : dimensione del vettore h dal backbone (default 512)
        move_emb_dim : dimensione dell'embedding delle mosse (default 128)
 
    Forward:
        h          : [batch, backbone_dim]
        move_embs  : [batch, n_mosse, move_emb_dim]
        move_mask  : [batch, n_mosse] bool, True = mossa reale, False = padding
                     (opzionale, serve solo se usi batching con padding)
 
    Returns:
        logits : [batch, n_mosse]   — score grezzi (usa per la loss)
        probs  : [batch, n_mosse]   — softmax sui soli slot reali
    """
    def __init__(self, backbone_dim: int = 512, move_emb_dim: int = 128):
        super().__init__()
        # Proiettiamo h nello stesso spazio degli embedding delle mosse
        self.query_proj = nn.Linear(backbone_dim, move_emb_dim)
        self.scale      = math.sqrt(move_emb_dim)
 
    def forward(
        self,
        h: torch.Tensor,
        move_embs: torch.Tensor,
        move_mask: torch.Tensor | None = None,
    ):
        # query: [batch, 1, move_emb_dim]
        query = self.query_proj(h).unsqueeze(1)
 
        # scores: [batch, n_mosse]
        scores = (query * move_embs).sum(dim=-1) / self.scale
 
        # Azzera il padding prima del softmax
        if move_mask is not None:
            scores = scores.masked_fill(~move_mask, float('-inf'))
 
        probs = torch.softmax(scores, dim=-1)
        return scores, probs

class ValueHead(nn.Module):
    """
    Input:  [batch, 512]
    Output: [batch, 1]  — valore della posizione in [-1, 1]
    """
    def __init__(self, backbone_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
 
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)

class JellyFishPointer(nn.Module):
    """
    Architettura completa con pointer attention per la policy.
 
    Input:
        board      : [batch, 13, 8, 8]
        moves      : [batch, n_mosse, 46]   — mosse legali encodate
        move_mask  : [batch, n_mosse] bool  — True = slot reale (opzionale)
 
    Output:
        logits     : [batch, n_mosse]       — per CrossEntropyLoss
        probs      : [batch, n_mosse]       — distribuzione policy
        value      : [batch, 1]             — valore posizione
    """
    def __init__(
        self,
        backbone_layers: list[int] = [2, 2, 2, 2],
        move_emb_dim:    int       = 128,
    ):
        super().__init__()
        self.backbone     = ChessBackbone(layers=backbone_layers)
        self.move_encoder = MoveEncoder(move_emb_dim=move_emb_dim)
        self.policy_head  = PointerPolicyHead(backbone_dim=512, move_emb_dim=move_emb_dim)
        self.value_head   = ValueHead(backbone_dim=512)
 
    def forward(
        self,
        board:     torch.Tensor,
        moves:     torch.Tensor,
        move_mask: torch.Tensor | None = None,
    ):
        h          = self.backbone(board)              # [batch, 512]
        move_embs  = self.move_encoder(moves)          # [batch, n_mosse, 128]
        logits, probs = self.policy_head(h, move_embs, move_mask)
        value      = self.value_head(h)                # [batch, 1]
        return logits, probs, value
    

class JellyFishPointerGCN(nn.Module):
    def __init__(self, move_emb_dim = 128):
        super().__init__()
        
        self.move_emb_dim = move_emb_dim

        self.GCN = PoolingChessGCN(input_dim=15,hidden_dim=self.move_emb_dim)
        self.move_encoder = MoveEncoder(move_emb_dim=self.move_emb_dim)
        self.policy_head  = PointerPolicyHead(backbone_dim=self.GCN.combined_dim, move_emb_dim=move_emb_dim)
        self.value_head   = ValueHead(backbone_dim=self.GCN.combined_dim)


    def forward(self,
                data:     torch.Tensor,
                moves:     torch.Tensor,
                move_mask: torch.Tensor | None = None,):

        combined = self.GCN(data.x, data.edge_index, data.batch, data.global_features)

        move_embs = self.move_encoder(moves)
        logits, probs = self.policy_head(combined, move_embs, move_mask)
        value      = self.value_head(combined)

        return logits, probs, value
    
class ChessTransformerLayer(nn.Module):
    """
    Un layer transformer con:
        - MHA standard + bias cinematico additivo fisso sugli attention score
        - FFN (d_model → ffn_dim → d_model) con GELU
        - Pre-LayerNorm (più stabile del post-LN per training lungo)

    Il bias cinematico `matrices_moves` è (n_heads, 64, 64).
    Viene pesato da `bias_scale` (scalare learnable, init=0.0):

        scores = QK^T / sqrt(d_head) + bias_scale * matrices_moves

    Inizializzare bias_scale=0 significa che il modello parte identico
    a MHA standard e impara gradualmente quanto fidarsi del prior cinematico.
    Questo risolve il problema del lambda che esplodeva nelle versioni precedenti
    (che erano init=1 e learnable per testa separatamente).
    """

    def __init__(
        self,
        d_model:   int   = 256,
        n_heads:   int   = 8,
        ffn_dim:   int   = 512,
        dropout:   float = 0.1,
        matrices_moves: torch.Tensor = None,  # (n_heads, 64, 64)
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        # Pre-LN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Proiezioni MHA
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Bias cinematico: fisso, un vettore learnable (uno per testa) per pesarlo
        if matrices_moves is not None:
            # (n_heads, 64, 64) — broadcastato su batch
            self.register_buffer('kin_bias', matrices_moves)
            # Un valore per testa, tutti a 0: parte neutro, ogni testa impara
            # autonomamente quanto fidarsi del proprio prior cinematico
            self.bias_scale = nn.Parameter(torch.zeros(n_heads))
        else:
            self.kin_bias   = None
            self.bias_scale = None

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 64, d_model)
        Returns:
            x: (batch, 64, d_model)
        """
        B, S, _ = x.shape  # S = 64

        # --- MHA con Pre-LN ---
        residual = x
        x = self.norm1(x)

        Q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: (B, n_heads, 64, head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: (B, n_heads, 64, 64)

        # Bias cinematico additivo
        if self.kin_bias is not None:
            # kin_bias: (n_heads, 64, 64) → broadcast su batch
            scores = scores + self.bias_scale.view(1, self.n_heads, 1, 1) * self.kin_bias.unsqueeze(0)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)                                   # (B, n_heads, 64, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)  # (B, 64, d_model)
        out = self.o_proj(out)

        x = residual + out

        # --- FFN con Pre-LN ---
        x = x + self.ffn(self.norm2(x))

        return x
    
class BoardEmbedding(nn.Module):
    """
    Proietta le feature di ogni casella in d_model.

    Input:  (batch, 64, input_dim)   input_dim=13 (one-hot 12 pezzi + turno)
    Output: (batch, 64, d_model)
    """
    def __init__(self, input_dim: int = 13, d_model: int = 256):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))
    
class ChessBoardEncoder(nn.Module):
    """
    Stack di N ChessTransformerLayer.

    Input:  (batch, 64, input_dim)
    Output: board_ctx (batch, d_model)  — mean pool su tutti i 64 token
    """
    def __init__(
        self,
        input_dim:      int   = 13,
        d_model:        int   = 256,
        n_heads:        int   = 8,
        n_layers:       int   = 4,
        ffn_dim:        int   = 512,
        dropout:        float = 0.1,
        kin_bias_path:  str   = 'attention_based_matrix_64x64.pt',
    ):
        super().__init__()
        self.d_model = d_model

        # Carica le matrici cinematiche una sola volta
        try:
            matrices = torch.load(kin_bias_path, weights_only=True)  # (7, 64, 64)
        except Exception:
            matrices = None
            print(f"[ChessBoardEncoder] Attenzione: impossibile caricare {kin_bias_path}. "
                  "Il bias cinematico sarà disabilitato.")

        # Se n_heads != 7 interpola / replica le matrici per adattarle
        # In questo caso default n_heads=8, matrici=7 → padding con zeros sull'ultima testa
        if matrices is not None and matrices.shape[0] != n_heads:
            matrices = _adapt_kin_matrices(matrices, n_heads)

        self.embedding = BoardEmbedding(input_dim, d_model)

        self.layers = nn.ModuleList([
            ChessTransformerLayer(
                d_model        = d_model,
                n_heads        = n_heads,
                ffn_dim        = ffn_dim,
                dropout        = dropout,
                matrices_moves = matrices,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 64, input_dim)  oppure (batch, 13, 8, 8) → reshape interno
        Returns:
            board_ctx: (batch, d_model)
        """
        # Supporta anche input (batch, 13, 8, 8) come da encode_board() originale
        if x.dim() == 4:
            B = x.shape[0]
            x = x.view(B, 13, 64).permute(0, 2, 1)  # (B, 64, 13)

        x = self.embedding(x)          # (B, 64, d_model)

        for layer in self.layers:
            x = layer(x)               # (B, 64, d_model)

        x = self.final_norm(x)

        # Mean pool → contesto globale della posizione
        board_ctx = x.mean(dim=1)      # (B, d_model)
        return board_ctx


def _adapt_kin_matrices(M: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Adatta le matrici cinematiche (7, 64, 64) a (n_heads, 64, 64).
    - Se n_heads < 7: tronca
    - Se n_heads > 7: padding con zeros per le teste extra
    Il padding con zeros significa che le teste extra partono senza prior
    cinematico e imparano liberamente (utile con n_heads=8).
    """
    orig_heads = M.shape[0]
    if n_heads <= orig_heads:
        return M[:n_heads]
    else:
        pad = torch.zeros(n_heads - orig_heads, 64, 64, dtype=M.dtype)
        return torch.cat([M, pad], dim=0)

    
class JellyFishPointerTransformer(nn.Module):
    """
    Transformer encoder + Pointer Network per scacchi.

    Forward:
        data      : torch_geometric.data.Data  con .x (B*64, 15) e .batch
                    OPPURE tensore (B, 13, 8, 8) o (B, 64, 13)
        moves     : (B, N_mosse, 46)
        move_mask : (B, N_mosse) bool

    Returns:
        logits : (B, N_mosse)
        probs  : (B, N_mosse)
        value  : (B, 1)

    Parametri consigliati per ottimizzazione:
        - bias_scale (uno per layer): lr=1e-4
        - tutto il resto:             lr=1e-3
    
    Esempio:
        optimizer = torch.optim.AdamW([
            {'params': [p for n,p in model.named_parameters()
                        if 'bias_scale' not in n], 'lr': 1e-3},
            {'params': [p for n,p in model.named_parameters()
                        if 'bias_scale' in n],     'lr': 1e-4},
        ], weight_decay=1e-4)
    """

    def __init__(
        self,
        input_dim:     int   = 13,
        d_model:       int   = 256,
        n_heads:       int   = 8,
        n_layers:      int   = 4,
        ffn_dim:       int   = 512,
        dropout:       float = 0.1,
        move_emb_dim:  int   = 256,   # ← aggiungi questo parametro
        kin_bias_path: str   = 'attention_based_matrix_64x64.pt',
    ):
        super().__init__()

        self.encoder    = ChessBoardEncoder(
            input_dim     = input_dim,
            d_model       = d_model,
            n_heads       = n_heads,
            n_layers      = n_layers,
            ffn_dim       = ffn_dim,
            dropout       = dropout,
            kin_bias_path = kin_bias_path,
        )
        self.move_encoder = MoveEncoder(d_model)
        self.policy_head  = PointerPolicyHead(backbone_dim=d_model, move_emb_dim=move_emb_dim)
        self.value_head   = ValueHead(d_model)

    def forward(
        self,
        x:         torch.Tensor,
        moves:     torch.Tensor,
        move_mask: torch.Tensor | None = None,
    ):
        board_ctx  = self.encoder(x)                              # (B, d_model)
        move_embs  = self.move_encoder(moves)                     # (B, N_mosse, d_model)
        logits, probs = self.policy_head(board_ctx, move_embs, move_mask)
        value      = self.value_head(board_ctx)                   # (B, 1)

        return logits, probs, value
    
    def get_optimizer(
        self,
        lr_main:  float = 1e-3,
        lr_bias:  float = 1e-4,
        wd:       float = 1e-4,
    ) -> torch.optim.Optimizer:
        """
        Ritorna un AdamW con lr separato per bias_scale.
        Chiama questo invece di costruire l'optimizer a mano.
        """
        bias_params = [p for n, p in self.named_parameters() if 'bias_scale' in n]
        main_params = [p for n, p in self.named_parameters() if 'bias_scale' not in n]
        return torch.optim.AdamW([
            {'params': main_params, 'lr': lr_main, 'weight_decay': wd},
            {'params': bias_params, 'lr': lr_bias, 'weight_decay': 0.0},
        ])