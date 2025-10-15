import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super(GatedFusion, self).__init__()
        # Project concatenated features down to D
        self.gate_fc = nn.Linear(2 * d_model, d_model)

    def forward(self, x, y):
        """
        x: [B, T, D]
        y: [B, T, D]
        returns: [B, T, D]
        """
        # Concatenate along feature dim
        concat = torch.cat([x, y], dim=-1)  # [B, T, 2D]

        # Compute gating weights
        gate = torch.sigmoid(self.gate_fc(concat))  # [B, T, D]

        # Fuse adaptively
        z = gate * x + (1 - gate) * y
        return z
