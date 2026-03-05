"""
vqvae_chess.py
==============
VQ-VAE convoluzionale per posizioni di scacchi.

Input:  tensore (B, 13, 8, 8)
         - piani 0-11: pezzi bianchi/neri (binari)
         - piano  12 : turno (0 o 1, costante su tutta la tavola)

Architettura
------------
  Encoder  → z_e  (B, D, H', W')   con D=dim_latente, H'=W'=2
  VQ Layer → z_q  (B, D, H', W')   codebook di K=512 vettori
  Decoder  → x̂   (B, 13, 8, 8)

Loss
----
  1. Ricostruzione separata per pezzi e turno:
       - Piani 0-11 (pezzi): Focal Loss pesata per classe
         I piani sono quasi tutti zero → bisogna pesare gli 1
       - Piano 12  (turno):  BCE standard
  2. VQ commitment loss  (β * ||z_e - sg[z_q]||²)
  3. Codebook loss       (||sg[z_e] - z_q||²)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Focal Loss per piani binari sparsi
# ─────────────────────────────────────────────

class CategoricalBoardLoss(nn.Module):
    """
    Cross-entropy categorica per la ricostruzione della scacchiera.

    Ogni casella è una classificazione a 13 classi:
        classe 0   = casella vuota
        classi 1-12 = i 12 tipi di pezzi (stessa mappatura del tensore input)

    Questo garantisce strutturalmente che ogni casella abbia al massimo un pezzo,
    eliminando il problema delle posizioni generate con pezzi sovrapposti.

    empty_weight: peso della classe vuota (~97% delle caselle), deve essere basso
                  per non far dominare la classe vuota nell'ottimizzazione.
    """
    def __init__(self, empty_weight: float = 0.03):
        super().__init__()
        self.empty_weight = empty_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, 13, 8, 8)  — classe 0=vuoto, 1-12=pezzi
        targets: (B, 12, 8, 8)  — piani binari originali (0=assente, 1=presente)
        """
        # Converte targets binari (B, 12, 8, 8) → indici categorici (B, 8, 8)
        piece_idx  = targets.argmax(dim=1) + 1          # (B, 8, 8), valori 1-12
        is_empty   = targets.sum(dim=1) == 0            # (B, 8, 8)
        target_idx = torch.where(is_empty,
                                 torch.zeros_like(piece_idx),
                                 piece_idx)              # (B, 8, 8), valori 0-12

        weight = torch.ones(13, device=logits.device)
        weight[0] = self.empty_weight

        return F.cross_entropy(logits, target_idx, weight=weight)


# ─────────────────────────────────────────────
# Vector Quantization Layer
# ─────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    """
    Straight-through VQ layer.

    K   : numero di vettori nel codebook (512)
    D   : dimensione di ogni vettore
    beta: peso della commitment loss

    Dettagli implementativi:
    - EMA update del codebook (più stabile del gradient update classico)
    - Codebook reset: se un vettore non viene usato per N step viene reinizializzato
    """
    def __init__(self, K: int = 512, D: int = 256, beta: float = 0.25,
                 ema_decay: float = 0.9, reset_threshold: int = 1):
        super().__init__()
        self.K = K
        self.D = D
        self.beta = beta

        # Pre-quantization projection: separa spazio encoder da spazio codebook.
        # Questo è il fix chiave: il decoder non può più bypassare il VQ perché
        # il gradiente deve passare obbligatoriamente attraverso questa proiezione.
        self.pre_proj  = nn.Linear(D, D)
        self.post_proj = nn.Linear(D, D)

        # Codebook
        self.embedding = nn.Embedding(K, D)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)

        # EMA con decay basso (0.9) per rispondere velocemente
        self.ema_decay = ema_decay
        self.register_buffer("ema_cluster_size", torch.ones(K) * (1.0 / K))
        self.register_buffer("ema_embed_sum",    self.embedding.weight.data.clone())

        # Contatore utilizzi per il reset (reset_threshold=1: resetta ogni step inutilizzato)
        self.register_buffer("usage_count", torch.zeros(K, dtype=torch.long))
        self.reset_threshold = reset_threshold

    def forward(self, z_e: torch.Tensor):
        """
        z_e: (B, D, H, W)
        Returns:
            z_q          : (B, D, H, W)  quantizzato (straight-through gradient)
            vq_loss      : scalare
            commitment   : scalare
            encoding_indices: (B*H*W,)
        """
        B, D, H, W = z_e.shape
        z_e = z_e.float()

        # Pre-projection: mappa z_e nello spazio del codebook
        flat_in = z_e.permute(0, 2, 3, 1).reshape(-1, D)  # (N, D)
        flat = self.pre_proj(flat_in)                       # (N, D)
        # L2-normalize per stabilità delle distanze
        flat = F.normalize(flat, dim=-1)
        emb  = F.normalize(self.embedding.weight, dim=-1)  # (K, D)

        # Distanze coseno → più stabili numericamente di L2 con spazi grandi
        dist = -(flat @ emb.t())   # (N, K), negativo perché vogliamo il massimo coseno
        encoding_indices = dist.argmin(dim=1)   # (N,)

        # Aggiorna contatori utilizzo
        self.usage_count.zero_()  # reset ogni step, contiamo uso nel batch corrente
        self.usage_count.index_add_(0, encoding_indices,
                                    torch.ones_like(encoding_indices))

        # EMA update del codebook
        if self.training:
            one_hot = F.one_hot(encoding_indices, self.K).float()  # (N, K)
            self.ema_cluster_size.mul_(self.ema_decay).add_(
                one_hot.sum(0) * (1 - self.ema_decay))
            self.ema_embed_sum.mul_(self.ema_decay).add_(
                (one_hot.t() @ flat.detach()) * (1 - self.ema_decay))
            # Normalizza (Laplace smoothing per evitare divisione per zero)
            new_emb = self.ema_embed_sum / (self.ema_cluster_size.unsqueeze(1) + 1e-5)
            self.embedding.weight.data = F.normalize(new_emb, dim=-1)

        # Codebook lookup
        z_q_flat = emb[encoding_indices]                            # (N, D) normalizzato
        z_q_flat = self.post_proj(z_q_flat)                        # (N, D) riproiettato
        z_q = z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)   # (B, D, H, W)

        # Commitment loss: l'encoder deve avvicinarsi al codebook
        # Usiamo flat_in (prima della normalizzazione) per un segnale di gradiente più ricco
        z_q_for_commit = z_q.detach()
        commitment = F.mse_loss(z_e, z_q_for_commit)

        total_vq = self.beta * commitment

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, total_vq, commitment, encoding_indices

    def reset_dead_codes(self, z_e_flat: torch.Tensor):
        """
        Reinizializza i vettori morti con campioni casuali da z_e (normalizzati).
        Chiamare ogni N step di training.
        """
        dead = (self.usage_count == 0).nonzero(as_tuple=True)[0]
        if len(dead) == 0:
            return 0
        n_dead = len(dead)
        # Proietta e normalizza come nel forward
        with torch.no_grad():
            flat = self.pre_proj(z_e_flat.float())
            flat = F.normalize(flat, dim=-1)
        n_samples = flat.size(0)
        # Se abbiamo meno campioni che vettori morti, campiona con rimpiazzo
        if n_samples >= n_dead:
            idx = torch.randperm(n_samples, device=flat.device)[:n_dead]
        else:
            idx = torch.randint(0, n_samples, (n_dead,), device=flat.device)
        self.embedding.weight.data[dead] = flat[idx].detach()
        self.ema_embed_sum[dead] = flat[idx].detach()
        self.ema_cluster_size[dead] = 1.0 / self.K
        return n_dead


# ─────────────────────────────────────────────
# Residual Block
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


# ─────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────

class ChessEncoder(nn.Module):
    """
    (B, 13, 8, 8) → (B, latent_dim, 2, 2)

    Strategia:
    - Piano 12 (turno) viene separato prima del processing
      e concatenato come feature globale
    - Stride 2 × 2 volte: 8→4→2
    """
    def __init__(self, latent_dim: int = 256, base_ch: int = 128):
        super().__init__()
        # Processa i 12 piani pezzi
        self.piece_stem = nn.Sequential(
            nn.Conv2d(12, base_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
        )
        # Turno → embedding scalare proiettato su tutti i canali
        self.turn_proj = nn.Linear(1, base_ch)

        # Downsampling 8→4
        self.down1 = nn.Sequential(
            ResBlock(base_ch),
            ResBlock(base_ch),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_ch * 2),
            nn.GELU(),
        )
        # Downsampling 4→2
        self.down2 = nn.Sequential(
            ResBlock(base_ch * 2),
            ResBlock(base_ch * 2),
            nn.Conv2d(base_ch * 2, latent_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, latent_dim),
            nn.GELU(),
        )
        self.out_proj = nn.Conv2d(latent_dim, latent_dim, 1)

    def forward(self, x: torch.Tensor):
        pieces = x[:, :12]          # (B, 12, 8, 8)
        turn   = x[:, 12:13, 0, 0]  # (B, 1)  — costante su tutto il piano

        h = self.piece_stem(pieces)  # (B, base_ch, 8, 8)

        # Aggiunge info turno come bias spaziale
        t = self.turn_proj(turn).unsqueeze(-1).unsqueeze(-1)  # (B, base_ch, 1, 1)
        h = h + t

        h = self.down1(h)   # (B, base_ch*2, 4, 4)
        h = self.down2(h)   # (B, latent_dim, 2, 2)
        return self.out_proj(h)


# ─────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────

class ChessDecoder(nn.Module):
    """
    (B, latent_dim, 2, 2) → (B, 13, 8, 8)

    Output:
    - logits piani 0-11 (pezzi):  usati con Focal Loss / sigmoid
    - logit piano 12   (turno):   usato con BCE / sigmoid
    """
    def __init__(self, latent_dim: int = 256, base_ch: int = 128):
        super().__init__()
        # Upsampling 2→4
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_ch * 2, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_ch * 2),
            nn.GELU(),
            ResBlock(base_ch * 2),
            ResBlock(base_ch * 2),
        )
        # Upsampling 4→8
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            ResBlock(base_ch),
            ResBlock(base_ch),
        )
        # Testa pezzi: logit categorici per 13 classi (classe 0=vuoto, 1-12=pezzi)
        # Con softmax/argmax garantisce strutturalmente una sola classe per casella.
        self.piece_head = nn.Conv2d(base_ch, 13, 1)
        # Testa turno: un singolo logit medio
        self.turn_head  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch, 1),
        )

    def forward(self, z_q: torch.Tensor):
        h = self.up1(z_q)   # (B, base_ch*2, 4, 4)
        h = self.up2(h)     # (B, base_ch,   8, 8)

        # (B, 13, 8, 8): logit categorici, classe 0=vuoto, 1-12=pezzi
        piece_logits = self.piece_head(h)

        turn_logit = self.turn_head(h).view(-1, 1)
        turn_plane = turn_logit.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8)

        # (B, 14, 8, 8): piani 0-12 = logit pezzi categorici, piano 13 = turno
        return torch.cat([piece_logits, turn_plane], dim=1)


# ─────────────────────────────────────────────
# VQ-VAE completo
# ─────────────────────────────────────────────

class ChessVQVAE(nn.Module):
    """
    VQ-VAE per posizioni di scacchi.

    Parametri principali:
        latent_dim   : dimensione dei vettori nel codebook (default 256)
        num_embeddings: K = 512 vettori nel codebook
        base_ch      : canali base dell'encoder/decoder
        beta         : peso della commitment loss
        focal_alpha  : peso classe positiva nella focal loss
        focal_gamma  : focusing parameter focal loss
    """
    def __init__(
        self,
        latent_dim:      int   = 256,
        num_embeddings:  int   = 512,
        base_ch:         int   = 128,
        beta:            float = 0.25,
        focal_alpha:     float = 0.97,  # ~97% delle celle sono vuote, serve peso alto
        focal_gamma:     float = 2.0,
        aux_weight:      float = 0.5,   # peso loss ausiliaria valutazione
    ):
        super().__init__()
        self.encoder  = ChessEncoder(latent_dim, base_ch)
        self.vq       = VectorQuantizer(num_embeddings, latent_dim, beta)
        self.decoder  = ChessDecoder(latent_dim, base_ch)

        # Testa ausiliaria: predice la valutazione posizionale da z_q
        # Forza lo spazio latente a organizzarsi semanticamente
        self.eval_head = nn.Sequential(
            nn.Linear(latent_dim * 4, 128),   # 4 = H'*W' = 2*2
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh(),                         # output in [-1, 1] come le label
        )
        self.aux_weight = aux_weight

        self.piece_loss_fn = CategoricalBoardLoss(empty_weight=0.03)
        self.bce           = nn.BCEWithLogitsLoss()
        self.mse           = nn.MSELoss()

    def forward(self, x: torch.Tensor, aux_target: torch.Tensor = None):
        """
        x:          (B, 13, 8, 8)
        aux_target: (B,) valutazione posizionale in [-1, 1], opzionale
        Returns dict con tutte le loss e le quantità utili per il logging.
        """
        z_e = self.encoder(x)
        z_q, vq_loss, commitment, indices = self.vq(z_e)
        x_recon = self.decoder(z_q)

        # Loss pezzi: cross-entropy categorica su 13 classi (0=vuoto, 1-12=pezzi)
        # x_recon[:, :13] = logit categorici decoder, x[:, :12] = piani binari target
        piece_loss = self.piece_loss_fn(x_recon[:, :13], x[:, :12])

        # Loss turno: piano 13 del decoder vs piano 12 dell'input
        turn_loss  = self.bce(x_recon[:, 13:14, 0, 0],
                              x[:, 12:13, 0, 0])

        recon_loss = piece_loss + 0.1 * turn_loss
        total_loss = recon_loss + vq_loss

        # Loss ausiliaria: predice la valutazione posizionale da z_q
        # Forza lo spazio latente a organizzarsi semanticamente
        aux_loss = torch.tensor(0.0, device=x.device)
        eval_pred = None
        if aux_target is not None and self.aux_weight > 0:
            z_q_flat   = z_q.reshape(z_q.size(0), -1).float()  # (B, D*4)
            eval_pred  = self.eval_head(z_q_flat).squeeze(1)    # (B,)
            aux_loss   = self.mse(eval_pred, aux_target.float())
            total_loss = total_loss + self.aux_weight * aux_loss

        return {
            "loss":        total_loss,
            "recon_loss":  recon_loss,
            "piece_loss":  piece_loss,
            "turn_loss":   turn_loss,
            "vq_loss":     vq_loss,
            "aux_loss":    aux_loss,
            "commitment":  commitment,
            "x_recon":     x_recon,
            "indices":     indices,
            "z_e":         z_e,
            "eval_pred":   eval_pred,
        }

    def encode(self, x: torch.Tensor):
        """Restituisce gli indici del codebook per ogni posizione."""
        z_e = self.encoder(x)
        _, _, _, indices = self.vq(z_e)
        return indices

    def decode_indices(self, indices: torch.Tensor, spatial_shape=(2, 2)):
        """Ricostruisce posizioni a partire dagli indici (per generazione)."""
        B_HW = indices.shape[0]
        H, W = spatial_shape
        B = B_HW // (H * W)
        z_q = self.vq.embedding(indices).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return torch.sigmoid(self.decoder(z_q))

    def codebook_usage(self) -> float:
        """Percentuale di vettori del codebook usati almeno una volta."""
        return (self.vq.usage_count > 0).float().mean().item()


# ─────────────────────────────────────────────
# Sanity check veloce
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = ChessVQVAE(latent_dim=256, num_embeddings=512, base_ch=128)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri totali: {total_params:,}")

    x = torch.zeros(4, 13, 8, 8)
    # Piazza qualche pezzo a caso per test
    x[:, 0, 6, :] = 1   # pedoni bianchi
    x[:, 6, 1, :] = 1   # pedoni neri
    x[:, 12]      = 1   # turno bianco

    out = model(x)
    print(f"Loss totale   : {out['loss'].item():.4f}")
    print(f"Recon loss    : {out['recon_loss'].item():.4f}")
    print(f"VQ loss       : {out['vq_loss'].item():.4f}")
    print(f"Commitment    : {out['commitment'].item():.4f}")
    print(f"Shape output  : {out['x_recon'].shape}")
    print(f"Indici shape  : {out['indices'].shape}")
