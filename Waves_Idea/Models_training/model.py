"""
model.py
---------
Lightweight dual-head ResNet for chess.
Predicts policy (move probabilities) and value (position evaluation).

Input:  (B, C, 8, 8)  where C=13 (baseline) or C=16 (with influence fields)
Output: policy logits (B, 1968) + value (B, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ChessNet(nn.Module):
    """
    Dual-head ResNet for chess.

    Args:
        in_channels : 13 for baseline, 16 for model with influence fields
        n_filters   : number of conv filters (default 64)
        n_blocks    : number of residual blocks (default 4)
        n_moves     : policy output size (default 1968)
        value_lambda: weight of value loss relative to policy loss
    """

    def __init__(
        self,
        in_channels: int = 13,
        n_filters:   int = 64,
        n_blocks:    int = 4,
        n_moves:     int = 1968,
    ):
        super().__init__()

        self.in_channels = in_channels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.tower = nn.Sequential(
            *[ResidualBlock(n_filters) for _ in range(n_blocks)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(n_filters, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, n_moves),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(n_filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, legal_mask=None):
        """
        Args:
            x           : (B, C, 8, 8) board tensor
            legal_mask  : (B, 1968) bool tensor — illegal moves get -inf before softmax

        Returns:
            policy_logits : (B, 1968)
            value         : (B, 1)
        """
        x = self.stem(x)
        x = self.tower(x)

        policy = self.policy_head(x)
        value  = self.value_head(x)

        if legal_mask is not None:
            policy = policy.masked_fill(~legal_mask, float('-inf'))

        return policy, value

    def predict(self, x, legal_mask=None):
        """Inference: returns softmax policy probs and scalar value."""
        with torch.no_grad():
            policy, value = self.forward(x, legal_mask)
            policy_probs  = torch.softmax(policy, dim=-1)
        return policy_probs, value.squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class AlphaZeroLoss(nn.Module):
    """
    L = L_policy + lambda_value * L_value

    L_policy : cross-entropy on move (with legal mask already applied)
    L_value  : MSE between predicted value and normalized stockfish eval
    """

    def __init__(self, value_lambda: float = 1.0):
        super().__init__()
        self.value_lambda = value_lambda

    def forward(self, policy_logits, value_pred, move_targets, value_targets):
        """
        Args:
            policy_logits  : (B, 1968) — illegal moves already masked
            value_pred     : (B, 1)
            move_targets   : (B,) long — index of correct move
            value_targets  : (B,) float — normalized eval in [-1, 1]
        """
        # Filter out positions where move target is -1 (unknown)
        valid = move_targets >= 0
        
        policy_loss = torch.tensor(0.0, device=policy_logits.device)
        if valid.any():
            policy_loss = F.cross_entropy(
                policy_logits[valid],
                move_targets[valid],
            )

        value_loss = F.mse_loss(
            value_pred.squeeze(-1),
            value_targets,
        )

        total = policy_loss + self.value_lambda * value_loss

        return total, policy_loss, value_loss


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for in_ch, name in [(13, 'Baseline'), (16, 'With fields')]:
        model = ChessNet(in_channels=in_ch)
        params = model.count_parameters()

        x    = torch.randn(4, in_ch, 8, 8)
        mask = torch.ones(4, 1968, dtype=torch.bool)

        policy, value = model(x, mask)

        print(f'{name}: {params:,} parameters')
        print(f'  policy shape: {policy.shape}')
        print(f'  value  shape: {value.shape}')
        print()
