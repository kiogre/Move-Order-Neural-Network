import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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