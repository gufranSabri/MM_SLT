import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
from modules.mstcn import MSTCN
from swin.swin_transformer import SwinTransformer
import torchvision.models as vision_models
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, loss_weights=None,
            weight_norm=True, share_classifier=True,
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.loss_weights = loss_weights

        c2d_type, t_model = c2d_type.split("_")[0], c2d_type.split("_")[1]
        self.t_model = t_model
        self.c2d_type = c2d_type
        self.adapter_type = int(self.t_model.split("-")[1]) if "-" in self.t_model else 0

        print(f"c2d_type: {c2d_type}, t_model: {t_model}")
        in_hidden_size = 768
        
        if "swin" in c2d_type and "3d" not in c2d_type:
            swin_t_config = {
                "patch_size": [4, 4],
                "embed_dim": 96,
                "depths": [2, 2, 6, 2],  # Swin-T specific depth configuration
                "num_heads": [3, 6, 12, 24],  # Number of attention heads
                "window_size": [7, 7],  # Window size for local self-attention
                "mlp_ratio": 4.0,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "stochastic_depth_prob": 0.2,  # Higher stochastic depth than Swin-S
                "num_classes": 1000,  # Default for ImageNet, change if needed
            }

            swin_s_config = {
                "patch_size": [4, 4],
                "embed_dim": 96,
                "depths": [2, 2, 18, 2],  # Swin-S specific depth configuration
                "num_heads": [3, 6, 12, 24],  # Number of attention heads
                "window_size": [7, 7],  # Window size for local self-attention
                "mlp_ratio": 4.0,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "stochastic_depth_prob": 0.2,  # Higher stochastic depth than Swin-S
                "num_classes": 1000,  # Default for ImageNet, change if needed
            }

            swin_b_config = {
                "patch_size": [4, 4],
                "embed_dim": 128,
                "depths": [2, 2, 18, 2],  # Swin-B specific depth configuration
                "num_heads": [4, 8, 16, 32],  # Number of attention heads
                "window_size": [7, 7],  # Window size for local self-attention
                "mlp_ratio": 4.0,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "stochastic_depth_prob": 0.2,  # Higher stochastic depth than Swin-S
                "num_classes": 1000,  # Default for ImageNet, change if needed
            }

            configs = {
                "swins": swin_s_config,
                "swint": swin_t_config,
                "swinb": swin_b_config,
                "swinslora": swin_s_config,
                "swintlora": swin_t_config,
                "swinblora": swin_b_config,
            }
            models = {
                "swins": vision_models.swin_s,
                "swint": vision_models.swin_t,
                "swinb": vision_models.swin_b,
                "swinslora": vision_models.swin_s,
                "swintlora": vision_models.swin_t,
                "swinblora": vision_models.swin_b,
            }
            weights = {
                "swins": vision_models.Swin_S_Weights.IMAGENET1K_V1,
                "swint": vision_models.Swin_T_Weights.IMAGENET1K_V1,
                "swinb": vision_models.Swin_B_Weights.IMAGENET1K_V1,
                "swinslora": vision_models.Swin_S_Weights.IMAGENET1K_V1,
                "swintlora": vision_models.Swin_T_Weights.IMAGENET1K_V1,
                "swinblora": vision_models.Swin_B_Weights.IMAGENET1K_V1,
            }
            ins = {
                "swint": [96, 192, 384, 768],
                "swins": [96, 192, 384, 768],
                "swinb": [128, 256, 512, 1024],
                "swinslora": [96, 192, 384, 768],
                "swintlora": [96, 192, 384, 768],
                "swinblora": [128, 256, 512, 1024]
            }

            model_w = models[c2d_type](weights=weights[c2d_type])
            self.conv2d = SwinTransformer(**configs[c2d_type])
            self.conv2d.load_weights(model_w)
            self.conv2d.modify(adapter=self.adapter_type, ins = ins[c2d_type], lora=True if "lora" in c2d_type else False)
            del model_w

            # hidden_size = 768 if c2d_type in ["swint", "swins", "swintlora", "swinslora"] else 1024
            in_hidden_size = 768 if c2d_type in ["swint", "swins", "swintlora", "swinslora"] else 1024

            if "lora" in c2d_type:
                for k,v in self.conv2d.named_parameters():
                    if "temporal_adapter" not in k and "features" in k and "lora" not in k and "norm" not in k and "mlp" not in k:
                        v.requires_grad = False

            print("Swin model loaded")

        self.conv1d = MSTCN(input_size=in_hidden_size, hidden_size=hidden_size)

        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)

    def forward(self, x, len_x, label=None):
        if len(x.shape) == 5:
            framewise = self.conv2d(x.permute(0,2,1,3,4)) # framewise -> [2, 2304, 188] -> [B, D, T]
        else:
            framewise = x

        if type(framewise) == tuple:
            framewise, prems = framewise

        conv1d_outputs = self.conv1d(framewise, len_x)
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(conv1d_outputs['visual_feat'], lgt)["predictions"]
        
        return {
            "feat_len": lgt,
            "visual_features": conv1d_outputs['visual_feat'],
            "sequence_features": tm_outputs,
        }