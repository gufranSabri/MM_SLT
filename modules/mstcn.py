import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MSTCN(nn.Module):
    def __init__(self, input_size, hidden_size, use_bn=False, num_classes=-1):
        super(MSTCN, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Match the same kernel sequence logic for T reduction
        self.kernel_size = ['K5', 'P2', 'K5', 'P2']

        self.streams = nn.ModuleList()

        # Stream 0: Conv1x1 + MaxPool3x1
        self.streams.append(nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        ))

        # Stream 1: Conv1x1
        self.streams.append(nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=1),
            nn.ReLU(inplace=True)
        ))

        # Streams 2–5: Conv1x1 + Conv3x1 with dilation = 1, 2, 3, 4
        for d in [1, 2, 3, 4]:
            self.streams.append(nn.Sequential(
                nn.Conv1d(input_size, hidden_size, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=d, padding=d),
                nn.ReLU(inplace=True)
            ))

        # After concat
        self.post_conv1 = nn.Sequential(
            nn.Conv1d(hidden_size * 6, hidden_size, kernel_size=3),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.div(feat_len, 2, rounding_mode='floor')
            elif ks[0] == 'K':
                feat_len -= int(ks[1]) - 1
        return feat_len

    def forward(self, x, lgt):
        feats = [stream(x) for stream in self.streams]
        x = torch.cat(feats, dim=1)
        x = self.post_conv1(x)

        return {
            "visual_feat": x.permute(2, 0, 1),  # (T, B, C)
            "feat_len": self.update_lgt(lgt).cpu(),
        }