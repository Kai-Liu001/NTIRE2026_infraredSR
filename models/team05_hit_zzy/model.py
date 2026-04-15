import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


TEAM05_SCALE = 4
TEAM05_GUIDE_MAP_OPT = {
    "enabled": True,
    "image_channels": 3,
    "maps": ["edge", "heat"],
    "edge": {
        "type": "sobel",
        "ksize": 3,
        "blur_sigma": 0.0,
    },
    "heat": {
        "type": "intensity",
        "blur_sigma": 1.2,
        "clahe": False,
    },
}
TEAM05_MODEL_CONFIG = {
    "img_size": 64,
    "in_chans": 5,
    "out_chans": 3,
    "img_range": 1.0,
    "d_state": 16,
    "depths": [8, 8, 8, 8, 8, 8],
    "embed_dim": 180,
    "mlp_ratio": 2,
    "resi_connection": "1conv",
}


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr=False, compress_ratio=3, squeeze_factor=30):
        super().__init__()
        if is_light_sr:
            compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x):
        return self.cab(x)


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([proj.weight for proj in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([proj.weight for proj in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([proj.bias for proj in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        a = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        a_log = torch.log(a)
        if copies > 1:
            a_log = repeat(a_log, "d n -> r d n", r=copies)
            if merge:
                a_log = a_log.flatten(0, 1)
        a_log = nn.Parameter(a_log)
        a_log._no_weight_decay = True
        return a_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        d = torch.ones(d_inner, device=device)
        if copies > 1:
            d = repeat(d, "n -> r n", r=copies)
            if merge:
                d = d.flatten(0, 1)
        d = nn.Parameter(d)
        d._no_weight_decay = True
        return d

    def forward_core(self, x):
        batch, _, height, width = x.shape
        length = height * width
        num_dirs = 4

        x_hwwh = torch.stack(
            [x.view(batch, -1, length), torch.transpose(x, dim0=2, dim1=3).contiguous().view(batch, -1, length)],
            dim=1,
        ).view(batch, 2, -1, length)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(batch, num_dirs, -1, length), self.x_proj_weight)
        dts, bs, cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(batch, num_dirs, -1, length), self.dt_projs_weight)

        xs = xs.float().view(batch, -1, length)
        dts = dts.contiguous().float().view(batch, -1, length)
        bs = bs.float().view(batch, num_dirs, -1, length)
        cs = cs.float().view(batch, num_dirs, -1, length)
        ds = self.Ds.float().view(-1)
        as_ = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs,
            dts,
            as_,
            bs,
            cs,
            ds,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(batch, num_dirs, -1, length)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(batch, 2, -1, length)
        wh_y = torch.transpose(out_y[:, 1].view(batch, -1, width, height), dim0=2, dim1=3).contiguous().view(batch, -1, length)
        invwh_y = torch.transpose(inv_y[:, 1].view(batch, -1, width, height), dim0=2, dim1=3).contiguous().view(batch, -1, length)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x, **kwargs):
        batch, height, width, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(batch, height, width, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim=0,
        drop_path=0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate=0.0,
        d_state=16,
        expand=2.0,
        is_light_sr=False,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim, is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        batch, _, channels = input.shape
        input = input.view(batch, *x_size, channels).contiguous()
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        return x.view(batch, -1, channels).contiguous()


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        drop_path=0.0,
        d_state=16,
        mlp_ratio=2.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        is_light_sr=False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=nn.LayerNorm,
                    attn_drop_rate=0,
                    d_state=d_state,
                    expand=mlp_ratio,
                    input_resolution=input_resolution,
                    is_light_sr=is_light_sr,
                )
            )

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                def _forward(inp):
                    return blk(inp, x_size)
                x = checkpoint.checkpoint(_forward, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    min_height = min(x1.size(2), x2.size(2), x3.size(2), x4.size(2))
    min_width = min(x1.size(3), x2.size(3), x3.size(3), x4.size(3))

    x1 = x1[:, :, :min_height, :min_width]
    x2 = x2[:, :, :min_height, :min_width]
    x3 = x3[:, :, :min_height, :min_width]
    x4 = x4[:, :, :min_height, :min_width]

    x_ll = x1 + x2 + x3 + x4
    x_hl = -x1 - x2 + x3 + x4
    x_lh = -x1 + x2 - x3 + x4
    x_hh = x1 - x2 - x3 + x4
    return x_ll, x_hl, x_lh, x_hh


class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class ChannelAttentionModified(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        mid_channels = max(1, in_channels // reduction_ratio)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(max_out)


class WaveletAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.dwt = DWT()
        self.channel_attention = ChannelAttentionModified(in_channels, reduction_ratio)

    def forward(self, x):
        c_a, c_h, c_v, c_d = self.dwt(x)
        c_a = F.interpolate(c_a, scale_factor=2, mode="bicubic", align_corners=False)
        c_h = F.interpolate(c_h, scale_factor=2, mode="bicubic", align_corners=False)
        c_v = F.interpolate(c_v, scale_factor=2, mode="bicubic", align_corners=False)
        c_d = F.interpolate(c_d, scale_factor=2, mode="bicubic", align_corners=False)
        return c_a, c_h, c_v, c_d


class FeatureMapping(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.silu(self.conv1(x)))


class FeatureModulation(nn.Module):
    def __init__(self, in_channels, wavelet_channels, scale_factor=1):
        super().__init__()
        self.mapping = FeatureMapping(in_channels * 4, wavelet_channels * scale_factor)
        self.scale_factor = scale_factor

    def forward(self, large_feature_map, detail_feature_map):
        modulation_params = self.mapping(detail_feature_map)
        if self.scale_factor > 1:
            modulation_params = F.interpolate(modulation_params, scale_factor=self.scale_factor, mode="bilinear")
        desired_size = (large_feature_map.size(2), large_feature_map.size(3))
        modulation_params = F.interpolate(modulation_params, size=desired_size, mode="bilinear", align_corners=False)
        return large_feature_map * modulation_params


class SmallScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv3x3(x)


class LargeScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwconv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.dwconv7x7(x))


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, wavelet_channels):
        super().__init__()
        mid_channels = in_channels
        wavelet_channels = in_channels
        self.small_scale_extractor = SmallScaleFeatureExtractor(in_channels, mid_channels)
        self.large_scale_extractor = LargeScaleFeatureExtractor(in_channels, in_channels)
        self.wavelet_attention = WaveletAttention(mid_channels)
        self.feature_modulation = FeatureModulation(mid_channels, wavelet_channels, scale_factor=1)
        self.fusion_conv = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1)

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        small_scale_features = self.small_scale_extractor(x)
        large_scale_features = self.large_scale_extractor(x)
        c_a, attn_c_h, attn_c_v, attn_c_d = self.wavelet_attention(small_scale_features)
        recombined_features = torch.cat((c_a, attn_c_h, attn_c_v, attn_c_d), dim=1)
        modulated_large_scale_features = self.feature_modulation(large_scale_features, recombined_features)
        combined_features = torch.cat([modulated_large_scale_features, small_scale_features], dim=1)
        return self.fusion_conv(combined_features)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        return x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])


class ResidualGroup(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        d_state=16,
        mlp_ratio=4.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=None,
        patch_size=None,
        resi_connection="1conv",
        is_light_sr=False,
    ):
        super().__init__()
        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state=d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr=is_light_sr,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unsupported resi_connection: {resi_connection}")

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        modules = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                modules.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                modules.append(nn.PixelShuffle(2))
        elif scale == 3:
            modules.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            modules.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"scale {scale} is not supported. Supported scales: 2^n and 3.")
        super().__init__(*modules)


def _build_channel_mean(num_channels, rgb_mean=(0.3317, 0.3317, 0.3317)):
    mean = torch.zeros(1, int(num_channels), 1, 1)
    if int(num_channels) >= 3:
        mean[:, :3, :, :] = torch.tensor(rgb_mean, dtype=torch.float32).view(1, 3, 1, 1)
    return mean


class Refinement(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, num_features=64):
        super().__init__()
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.refinement(x)


class DepthGuidedFusionModule(nn.Module):
    def __init__(self, in_channels=5, out_channels=64):
        super().__init__()
        self.in_channels = int(in_channels)
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
        )
        self.semantic_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, self.in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        conv_out = self.conv_block(x)
        attention_weights = self.semantic_attention(conv_out).unsqueeze(-1).unsqueeze(-1)
        return x * attention_weights


class _IRSRMambaMultiScaleGuidanceBase(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=5,
        out_chans=3,
        embed_dim=96,
        depths=(6, 6, 6, 6),
        drop_rate=0.0,
        d_state=16,
        mlp_ratio=2.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        img_range=1.0,
        resi_connection="1conv",
        **kwargs,
    ):
        super().__init__()
        self.num_in_ch = int(in_chans)
        self.num_out_ch = int(out_chans)
        num_feat = 64

        self.img_range = img_range
        self.register_buffer("input_mean", _build_channel_mean(self.num_in_ch))
        self.register_buffer("output_mean", _build_channel_mean(self.num_out_ch))

        self.featureFusionmodule = FeatureFusionModule(self.num_in_ch, self.num_in_ch, 4, self.num_in_ch)
        self.conv_first = nn.Conv2d(self.num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                d_state=d_state,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                is_light_sr=False,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unsupported resi_connection: {resi_connection}")

        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )

        self.up_step1 = Upsample(2, num_feat)
        self.up_step2 = Upsample(2, num_feat)
        self.conv_last = nn.Conv2d(num_feat, self.num_out_ch, 3, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        return self.patch_unembed(x, x_size)

    def preprocess_input(self, x):
        return x

    def _extract_feat(self, x):
        x = self.preprocess_input(x)
        input_mean = self.input_mean.to(dtype=x.dtype, device=x.device)
        x = (x - input_mean) * self.img_range
        x = self.featureFusionmodule(x)
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        return self.conv_before_upsample(x)

    def forward(self, x, scale: int):
        feat = self._extract_feat(x)
        if scale == 2:
            feat_up = self.up_step1(feat)
        elif scale == 4:
            feat_up = self.up_step2(self.up_step1(feat))
        else:
            raise ValueError(f"Unsupported scale={scale}. Expected 2/4.")

        out = self.conv_last(feat_up)
        output_mean = self.output_mean.to(dtype=out.dtype, device=out.device)
        return out / self.img_range + output_mean


class IRSRMambaMultiScaleConcatGuidance(_IRSRMambaMultiScaleGuidanceBase):
    pass


class IRSRMambaMultiScaleGuidedFusion(_IRSRMambaMultiScaleGuidanceBase):
    def __init__(
        self,
        image_in_chans=3,
        guide_in_chans=2,
        guide_refine_features=64,
        guided_fusion_channels=64,
        **kwargs,
    ):
        self.image_in_chans = int(image_in_chans)
        self.guide_in_chans = int(guide_in_chans)
        cfg_in_chans = int(kwargs.get("in_chans", self.image_in_chans + self.guide_in_chans))
        expected_in_chans = self.image_in_chans + self.guide_in_chans
        if cfg_in_chans != expected_in_chans:
            raise ValueError(
                f"`in_chans` must equal image_in_chans + guide_in_chans. "
                f"Got in_chans={cfg_in_chans}, image_in_chans={self.image_in_chans}, guide_in_chans={self.guide_in_chans}."
            )
        super().__init__(**kwargs)
        self.refinement = Refinement(
            in_channels=self.guide_in_chans,
            out_channels=self.guide_in_chans,
            num_features=int(guide_refine_features),
        )
        self.depth_guided_fusion = DepthGuidedFusionModule(
            in_channels=expected_in_chans,
            out_channels=int(guided_fusion_channels),
        )

    def preprocess_input(self, x):
        expected_channels = self.image_in_chans + self.guide_in_chans
        if x.size(1) != expected_channels:
            raise ValueError(f"Expected guided input with {expected_channels} channels, but got {x.size(1)}.")

        image = x[:, :self.image_in_chans, :, :]
        guides = x[:, self.image_in_chans:expected_channels, :, :]
        guides = self.refinement(guides)
        fused_input = torch.cat([image, guides], dim=1)
        return fused_input + self.depth_guided_fusion(fused_input)


class Team05IRSRMambaPlus(IRSRMambaMultiScaleConcatGuidance):
    def __init__(self, **overrides):
        config = dict(TEAM05_MODEL_CONFIG)
        config.update(overrides)
        super().__init__(**config)


def build_model(**overrides):
    return Team05IRSRMambaPlus(**overrides)


__all__ = [
    "TEAM05_SCALE",
    "TEAM05_GUIDE_MAP_OPT",
    "TEAM05_MODEL_CONFIG",
    "IRSRMambaMultiScaleConcatGuidance",
    "IRSRMambaMultiScaleGuidedFusion",
    "Team05IRSRMambaPlus",
    "build_model",
]
