import torch
import torch.nn as nn

class TimeCorrelatedConv(nn.Module):
    def __init__(self, c, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(
            in_channels=c + c,
            out_channels=c,
            kernel_size=kernel_size,
            padding=padding
        )
        self.activation = nn.GELU()

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x1, x2):
        """
        x1: [B, C1, T1]
        x2: [B, C2, T2]
        returns: [B, T1+T2, D]
        """
        x1 = x1.permute(0, 2, 1)  # [B, C, T1]
        x2 = x2.permute(0, 2, 1)  # [B, C, T2]

        B, C1, T1 = x1.shape
        B, C2, T2 = x2.shape

        # Expand to joint time grid
        x1_exp = x1.unsqueeze(-1).expand(-1, -1, -1, T2)  # [B, C, T1, T2]
        x2_exp = x2.unsqueeze(-2).expand(-1, -1, T1, -1)  # [B, C, T1, T2]

        # Concatenate along channels
        joint = torch.cat([x1_exp, x2_exp], dim=1)  # [B, C+C, T1, T2]

        # Apply 2D temporal convolution
        out = self.activation(self.conv2d(joint))  # [B, D, T1, T2]

        # Collapse along T2 and T1 respectively
        out_T1 = out.mean(dim=-1).permute(0, 2, 1)  # [B, T1, D]

        return out_T1 * self.alpha + x1.permute(0, 2, 1)  # [B, T1, D]


if __name__ == "__main__":
    B, T1, T2, D = 2, 10, 15, 768
    x1 = torch.randn(B, T1, D)
    x2 = torch.randn(B, T2, D)
    model = CrossTemporalConv(D)
    out = model(x1, x2)
    print(out.shape)  # Expected: [B, T1+T2, D]