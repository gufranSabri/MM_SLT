import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Temporal_LiftPool(nn.Module):
    def __init__(self, input_size, kernel_size=2):
        super(Temporal_LiftPool, self).__init__()
        self.kernel_size = kernel_size
        self.predictor = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, stride=1, padding=1, groups=input_size), 
            nn.ReLU(inplace=True),   
            nn.Conv1d(input_size, input_size, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),    
        )   

        self.updater = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, stride=1, padding=1, groups=input_size),
            nn.ReLU(inplace=True),   
            nn.Conv1d(input_size, input_size, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),    
        )
        
        self.predictor[2].weight.data.fill_(0.0)
        self.updater[2].weight.data.fill_(0.0)
        self.weight1 = Local_Weighting(input_size)
        self.weight2 = Local_Weighting(input_size)

    def forward(self, x):
        B, C, T= x.size()
        Xe = x[:,:,:T:self.kernel_size]
        Xo = x[:,:,1:T:self.kernel_size]
        d = Xo - self.predictor(Xe)
        s = Xe + self.updater(d)
        loss_u = torch.norm(s-Xo, p=2)
        loss_p = torch.norm(d, p=2)
        s = torch.cat((x[:,:,:0:self.kernel_size], s, x[:,:,T::self.kernel_size]),2)
        return self.weight1(s)+self.weight2(d), loss_u, loss_p

class Local_Weighting(nn.Module):
    def __init__(self, input_size ):
        super(Local_Weighting, self).__init__()
        self.conv = nn.Conv1d(input_size, input_size, kernel_size=5, stride=1, padding=2)
        self.insnorm = nn.InstanceNorm1d(input_size, affine=True)
        self.conv.weight.data.fill_(0.0)

    def forward(self, x):
        out = self.conv(x)
        return x + x*(F.sigmoid(self.insnorm(out))-0.5)
        

class MSTCN(nn.Module):
    def __init__(self, input_size, hidden_size, use_bn=False):
        super(MSTCN, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size

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
            nn.Conv1d(hidden_size * 6, hidden_size, kernel_size=5),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.tlp1 = Temporal_LiftPool(hidden_size, kernel_size=2)

        self.post_conv2 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.tlp2 = Temporal_LiftPool(hidden_size, kernel_size=2)


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
        x, loss_u1, loss_p1 = self.tlp1(x)

        x = self.post_conv2(x)
        x, loss_u2, loss_p2 = self.tlp2(x)

        return {
            "visual_feat": x.permute(2, 0, 1),  # (T, B, C)
            "feat_len": self.update_lgt(lgt).cpu(),
            "loss_LiftPool_u": loss_u1 + loss_u2,
            "loss_LiftPool_p": loss_p1 + loss_p2
        }

