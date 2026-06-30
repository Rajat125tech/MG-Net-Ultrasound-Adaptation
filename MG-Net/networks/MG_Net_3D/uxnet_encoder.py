import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
from monai.utils import ensure_tuple_rep, optional_import
from monai.networks.blocks import MLPBlock as Mlp

try:
    import natten
    if natten.HAS_LIBNATTEN:
        from natten import NeighborhoodAttention2D
    else:
        raise ImportError
except ImportError:
    class NeighborhoodAttention2D(nn.Module):
        def __init__(self, embed_dim, num_heads, kernel_size=3, dilation=1, qkv_bias=True):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.kernel_size = kernel_size
            self.dilation = dilation
            self.head_dim = embed_dim // num_heads
            self.scale = self.head_dim ** -0.5
            
            self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(embed_dim, embed_dim)

        def forward(self, x):
            B, H, W, C = x.shape
            qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(3)

            pad = (self.kernel_size - 1) // 2 * self.dilation
            
            k_padded = k.permute(0, 3, 4, 1, 2)
            v_padded = v.permute(0, 3, 4, 1, 2)
            
            k_padded = F.pad(k_padded, (pad, pad, pad, pad), mode='constant', value=0.0)
            v_padded = F.pad(v_padded, (pad, pad, pad, pad), mode='constant', value=0.0)
            
            k_neighs = []
            v_neighs = []
            for dy in range(self.kernel_size):
                for dx in range(self.kernel_size):
                    sy = dy * self.dilation
                    sx = dx * self.dilation
                    k_neighs.append(k_padded[..., sy:sy+H, sx:sx+W])
                    v_neighs.append(v_padded[..., sy:sy+H, sx:sx+W])
                    
            k_neigh = torch.stack(k_neighs, dim=-1)
            v_neigh = torch.stack(v_neighs, dim=-1)
            
            q_reg = q.permute(0, 3, 1, 2, 4).unsqueeze(-2)
            k_neigh_reg = k_neigh.permute(0, 1, 3, 4, 2, 5)
            
            attn = torch.matmul(q_reg, k_neigh_reg) * self.scale
            attn = F.softmax(attn, dim=-1)
            
            v_neigh_reg = v_neigh.permute(0, 1, 3, 4, 2, 5)
            out = torch.matmul(v_neigh_reg, attn.transpose(-1, -2)).squeeze(-1)
            
            out = out.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
            out = self.proj(out)
            return out

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if x.dim() == 4:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            else:
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

class ux_block(nn.Module):
    r""" ConvNeXt Block adapted for 2D.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.na2d_1 = NeighborhoodAttention2D(embed_dim=dim, num_heads=4, kernel_size=3, dilation=1)
        self.na2d_2 = NeighborhoodAttention2D(embed_dim=dim, num_heads=4, kernel_size=3, dilation=2)

        self.mlp = Mlp(dim, dim*4, act="GELU", dropout_rate=0.0, dropout_mode="swin")
       
    def forward(self, x):
        input = x

        # natten 1
        input_na = input.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        output_na = self.norm(input_na)
        output_na = self.na2d_1(output_na)
        output_na = output_na.permute(0, 3, 1, 2)
        output_na = self.drop_path(output_na) + input
        
        # mlp 1
        output_na = self.drop_path(self.mlp(self.norm(output_na.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)) + output_na

        # natten 2
        input_na_2 = output_na.permute(0, 2, 3, 1)
        output_na_2 = self.norm(input_na_2)
        output_na_2 = self.na2d_2(output_na_2)
        output_na_2 = output_na_2.permute(0, 3, 1, 2)
        output_na_2 = self.drop_path(output_na_2) + output_na
        
        # mlp 2
        output_na_2 = self.drop_path(self.mlp(self.norm(output_na_2.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)) + output_na_2

        x = output_na_2
        return x


class uxnet_conv(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ux_block(dim=dims[i], drop_path=dp_rates[cur + j],
                          layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x
