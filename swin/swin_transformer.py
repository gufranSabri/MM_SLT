import math
from functools import partial
from typing import Callable, List, Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from swin.misc import MLP, Permute
from swin.stochastic_depth import StochasticDepth

def _patch_merging_pad(x: torch.Tensor) -> torch.Tensor:
    H, W, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
    x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
    x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
    x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
    x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
    return x

torch.fx.wrap("_patch_merging_pad")

class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor):
        x = _patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        return x

def _get_relative_position_bias(
        relative_position_bias_table: torch.Tensor, 
        relative_position_index: torch.Tensor, 
        window_size: List[int]
    ) -> torch.Tensor:
    
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias

torch.fx.wrap("_get_relative_position_bias")


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, lora_alpha=1.0):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(out_features, r) * 0.01)
            self.lora_B = nn.Parameter(torch.randn(r, in_features) * 0.01)
            self.scaling = lora_alpha / r
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        if self.r > 0:
            return F.linear(x, self.lora_A @ self.lora_B) * self.scaling
        else:
            return torch.zeros_like(x)


def shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
    training: bool = True,
    qkv_lora_weight: Optional[Tensor] = None,
) -> Tensor:
    B, H, W, C = input.shape
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)

    if qkv_lora_weight is not None:
        qkv = qkv + qkv_lora_weight(x)

    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    if logit_scale is not None:
        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()

    return x

torch.fx.wrap("shifted_window_attention")

class ShiftedWindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size  # type: ignore[arg-type]
        )

    def lorify(self, r: int = 8, alpha: float = 1.0):
        self.qkv_lora = LoRALinear(self.dim, self.dim * 3, r=r, lora_alpha=alpha)
        print("Using LoRA in ShiftedWindowAttention with r =", r, "and alpha =", alpha)

    def forward(self, x: Tensor) -> Tensor:
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
            qkv_lora_weight=self.qkv_lora if hasattr(self, 'qkv_lora') else None,
        )

class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def lorify(self, r: int = 8, alpha: float = 1.0):
        self.attn.lorify(r=r, alpha=alpha)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class TemporalAdapterPE(nn.Module):
    def __init__(self, in_channels, adapter_channels=64, max_frames=64):
        super().__init__()
        self.in_channels = in_channels
        self.adapter_channels = adapter_channels
        self.T = None

        # Norm and projection
        self.norm1 = nn.LayerNorm(in_channels)
        self.down_proj = nn.Linear(in_channels, adapter_channels)

        # 🔸 Temporal Positional Encoding
        self.temporal_pos_emb = nn.Parameter(torch.zeros(max_frames, adapter_channels))  # (T, C)

        # Block 1
        self.block1_conv1x1 = nn.Conv3d(adapter_channels, adapter_channels, kernel_size=1, padding=0)
        self.block1_bn1 = nn.BatchNorm3d(adapter_channels)

        self.block1_conv3x3_1 = nn.Conv3d(adapter_channels, adapter_channels, kernel_size=3, padding=1)
        self.block1_bn2 = nn.BatchNorm3d(adapter_channels)
        self.block1_conv3x3_2 = nn.Conv3d(adapter_channels, adapter_channels, kernel_size=3, padding=1)
        self.block1_bn3 = nn.BatchNorm3d(adapter_channels)

        # Block 2
        self.block2_conv3x3_1 = nn.Conv3d(adapter_channels, adapter_channels, kernel_size=3, padding=1)
        self.block2_bn1 = nn.BatchNorm3d(adapter_channels)
        self.block2_conv3x3_2 = nn.Conv3d(adapter_channels, adapter_channels, kernel_size=3, padding=1)
        self.block2_bn2 = nn.BatchNorm3d(adapter_channels)

        self.norm2 = nn.LayerNorm(adapter_channels)
        self.up_proj = nn.Linear(adapter_channels, in_channels)

        # Bias and weight initialization
        nn.init.constant_(self.down_proj.bias, 0.)
        nn.init.constant_(self.up_proj.bias, 0.)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Input shape: (BT, H, W, C)
        BT, H, W, C = x.shape
        T = self.T
        B = BT // T

        x_id = x

        x = x.view(B, T, H, W, C)
        x = self.norm1(x)
        x = self.down_proj(x)

        # 🔸 Add Temporal Positional Encoding
        pos_emb = self.temporal_pos_emb[:T]  # (T, C)
        pos_emb = pos_emb[None, :, None, None, :]  # (1, T, 1, 1, C)
        x = x + pos_emb  # (B, T, H, W, C)

        # (B, T, H, W, C) -> (B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # Block 1
        stream1 = self.block1_bn1(self.block1_conv1x1(x))

        stream2 = self.block1_conv3x3_1(x)
        stream2 = self.block1_bn2(stream2)
        stream2 = F.gelu(stream2)
        stream2 = self.block1_conv3x3_2(stream2)
        stream2 = self.block1_bn3(stream2)

        x = stream1 + stream2

        # x = F.gelu(x)

        # Block 2
        residual = x
        x = self.block2_conv3x3_1(x)
        x = self.block2_bn1(x)
        x = F.gelu(x)
        x = self.block2_conv3x3_2(x)
        x = self.block2_bn2(x)

        x = x + residual
        x = F.gelu(x)

        # (B, C, T, H, W) -> (B, T, H, W, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm2(x)
        x = self.up_proj(x)

        x = x.view(BT, H, W, C)

        # x = self.weight_o * x_id + self.weight_t * x
        # return x

        return x_id + x


class ModifiedSwinLayer(nn.Module):
    def __init__(self, swin_layer, inC, adapter=3):
        super(ModifiedSwinLayer, self).__init__()
        self.swin_layer = swin_layer
        self.temporal_adapter = TemporalAdapterPE(inC, adapter_channels=64, max_frames=1000)
        self.adapter = adapter

    def forward(self, x):
        self.temporal_adapter.T = self.T
        x = self.swin_layer(x)
        x = self.temporal_adapter(x) + x
        
        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
    ):
        super().__init__()
        self.num_classes = num_classes

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        self.num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(self.num_features)
        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(self.num_features, 1000)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_weights(self, model_w):
        msg = self.load_state_dict(model_w.state_dict(), strict=False)
        self.head = nn.Linear(self.num_features, 512)
        print(msg)

    def modify(self, adapter=3, ins = [96, 192, 384, 768], lora=False):
        self.adapter = adapter
        if adapter == 0: return
        if adapter:
            # swin t and swin s
            self.features = nn.Sequential(
                self.features[0],
                ModifiedSwinLayer(self.features[1], inC=ins[0], adapter=adapter),
                self.features[2],
                ModifiedSwinLayer(self.features[3], inC=ins[1], adapter=adapter),
                self.features[4],
                ModifiedSwinLayer(self.features[5], inC=ins[2], adapter=adapter),
                self.features[6],
                ModifiedSwinLayer(self.features[7], inC=ins[3], adapter=adapter),
            )

            if lora:
                for i in range(len(self.features)):
                    if isinstance(self.features[i], ModifiedSwinLayer):
                        self.features[i].swin_layer[0].attn.lorify(r=8, alpha=1.0)
                        self.features[i].swin_layer[1].attn.lorify(r=8, alpha=1.0)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        B, T, C, H, W = x.shape
        prems = []


        if self.adapter != 0:
            self.features[1].T = T
            self.features[3].T = T
            self.features[5].T = T
            self.features[7].T = T

        x = x.reshape(B * T, C, H, W)

        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)
        x = self.features[3](x)
        prems.append(x)
        x = self.features[4](x)
        x = self.features[5](x)
        prems.append(x)
        x = self.features[6](x)
        x = self.features[7](x)
        prems.append(x)
        
        # x = self.features(x)

        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = x.reshape(B, T, self.num_features).permute(0, 2, 1)  # (B, D, T)

        return x, prems
    
