# model_sr_ir_context_x8_caswit_bicubic_residual.py
#
# Grayscale (1-band) x8-input SR with CASWiT backbone.
# Requirement:
# - Inputs x_hr/x_lr are 1-channel normalized tensors.
# - Output is 1-band prediction (residual over bicubic base).
#
# pred_1ch = base_1ch + residual_1ch
# where base_1ch is x_hr[:,0:1].

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from model.CASWiT_segformer import CASWiT  # expected in your repo


@dataclass
class SRModelCfg:
    model_name: str = "openmmlab/upernet-swin-base"
    backbone_init: str = "hf_pretrained"
    cross_attention_heads: int = 1
    fusion_mlp_ratio: float = 4.0
    fusion_drop_path: float = 0.1
    decoder_hidden_size: int = 256
    head_type: str = "hybrid"
    head_num_blocks: int = 10
    head_window_size: int = 8
    head_attention_heads: int = 8
    residual_zero_init: bool = True
    clamp_eval: bool = True


def _window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    B, C, H, W = x.shape
    pad_h = (window_size - (H % window_size)) % window_size
    pad_w = (window_size - (W % window_size)) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    Hp, Wp = x.shape[-2:]
    x = x.view(B, C, Hp // window_size, window_size, Wp // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
    return windows, (H, W, Hp, Wp)


def _window_reverse(windows: torch.Tensor, window_size: int, meta: Tuple[int, int, int, int], batch_size: int) -> torch.Tensor:
    H, W, Hp, Wp = meta
    x = windows.view(batch_size, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(batch_size, -1, Hp, Wp)
    return x[:, :, :H, :W]


class WindowAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int, window_size: int, shift_size: int = 0):
        super().__init__()
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))

        windows, meta = _window_partition(x, self.window_size)
        y = self.norm(windows)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = self.proj(y)
        y = _window_reverse(y, self.window_size, meta, B)

        if self.shift_size > 0:
            y = torch.roll(y, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))
        return y


class HybridAttnNAFBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
    ):
        super().__init__()
        self.local_branch = NAFBlock(channels, dw_expand=dw_expand, ffn_expand=ffn_expand, drop=0.0)
        self.global_branch = WindowAttention2d(
            channels=channels,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
        )
        self.global_scale = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.merge = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.merge_scale = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local_branch(x)
        global_feat = x + self.global_branch(x) * self.global_scale
        fused = self.merge(torch.cat([local, global_feat], dim=1))
        return local + fused * self.merge_scale


class PixelShuffleResidualUpsampler(nn.Module):
    def __init__(self, channels: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.GELU(),
            nn.PixelShuffle(2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )
        self.pred = nn.Conv2d(channels, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred(self.body(x))
        
class NAFBlock(nn.Module):
    """
    NAFNet-style block (image restoration).
    Paper: NAFNet (Chen et al.)
    This is a simplified version that works well as a residual head for SR.
    """
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, c)  # stable for SR/restoration
        self.pw1 = nn.Conv2d(c, c * dw_expand, 1, 1, 0)
        self.dw = nn.Conv2d(c * dw_expand, c * dw_expand, 3, 1, 1, groups=c * dw_expand)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * dw_expand, c * dw_expand, 1, 1, 0),
        )
        self.pw2 = nn.Conv2d(c * dw_expand, c, 1, 1, 0)

        self.norm2 = nn.GroupNorm(1, c)
        # ffn with gating: expand to 2*(c*ffn_expand), split, gate -> (c*ffn_expand), then project to c
        self.ffn1 = nn.Conv2d(c, 2 * c * ffn_expand, 1, 1, 0)
        self.ffn2 = nn.Conv2d(c * ffn_expand, c, 1, 1, 0)

        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        # learnable residual scalars (NAFNet trick)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # branch 1: depthwise conv + simplified channel attention
        y = self.norm1(x)
        y = self.pw1(y)
        y = self.dw(y)
        y = y * self.sca(y)
        y = self.pw2(y)
        y = self.drop(y)
        x = x + y * self.beta

        # branch 2: FFN (pointwise) with gating
        z = self.norm2(x)
        z = self.ffn1(z)  # [B, 2*(c*ffn_expand), H, W]
        a, b = torch.chunk(z, 2, dim=1)  # each [B, c*ffn_expand, H, W]
        z = a * torch.sigmoid(b)         # gated -> [B, c*ffn_expand, H, W]
        z = self.ffn2(z)                 # -> [B, c, H, W]
        z = self.drop(z)
        return x + z * self.gamma

class FusionBlock(nn.Module):
    """
    Local fusion block that refines concatenated fine/coarse features.
    """
    def __init__(self, c_fine: int, c_coarse: int, hidden: int):
        super().__init__()

        self.proj_fine = nn.Conv2d(c_fine, hidden, kernel_size=1)
        self.proj_coarse = nn.Conv2d(c_coarse, hidden, kernel_size=1)

        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.res = nn.Conv2d(hidden, hidden, kernel_size=1)

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        fine_p = self.proj_fine(fine)
        coarse_p = self.proj_coarse(coarse)

        coarse_up = F.interpolate(
            coarse_p,
            size=fine_p.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([fine_p, coarse_up], dim=1)
        x = self.fuse(x)

        # Keep the fine path dominant.
        return x + self.res(fine_p)


class HybridSRHead(nn.Module):
    """
    Progressive multi-scale fusion + hybrid restoration trunk + learned x2 output.
    """

    def __init__(
        self,
        in_channels: Tuple[int, int, int, int],
        hidden: int = 256,
        num_blocks: int = 10,
        out_ch: int = 1,
        window_size: int = 8,
        attn_heads: int = 8,
        zero_init: bool = True,
    ):
        super().__init__()
        c1, c2, c3, c4 = in_channels

        self.fuse43 = FusionBlock(c_fine=c3, c_coarse=c4, hidden=hidden)
        self.fuse32 = FusionBlock(c_fine=c2, c_coarse=hidden, hidden=hidden)
        self.fuse21 = FusionBlock(c_fine=c1, c_coarse=hidden, hidden=hidden)

        self.pre_trunk = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )

        blocks = []
        for idx in range(num_blocks):
            blocks.append(
                HybridAttnNAFBlock(
                    channels=hidden,
                    num_heads=attn_heads,
                    window_size=window_size,
                    shift_size=0 if idx % 2 == 0 else max(1, window_size // 2),
                )
            )
        self.trunk = nn.Sequential(*blocks)
        self.upsample = PixelShuffleResidualUpsampler(hidden, out_ch)

        if zero_init:
            nn.init.zeros_(self.upsample.pred.weight)
            if self.upsample.pred.bias is not None:
                nn.init.zeros_(self.upsample.pred.bias)

    def forward(
        self,
        feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        out_hw: Tuple[int, int],
    ) -> torch.Tensor:
        f1, f2, f3, f4 = feats

        x3 = self.fuse43(f3, f4)
        x2 = self.fuse32(f2, x3)
        x1 = self.fuse21(f1, x2)

        x = self.pre_trunk(x1)
        x = self.trunk(x)
        x = self.upsample(x)

        if x.shape[-2:] != out_hw:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return x


class CASWiT_SR_IR_X8_Residual(nn.Module):
    def __init__(self, cfg: SRModelCfg, pretrained_path: Optional[str] = None):
        super().__init__()
        self.input_channels = 1
        self.backbone = CASWiT(
            num_head_xa=int(cfg.cross_attention_heads),
            num_classes=12,
            model_name=str(cfg.model_name),
            mlp_ratio=float(cfg.fusion_mlp_ratio),
            drop_path=float(cfg.fusion_drop_path),
            backbone_init=str(cfg.backbone_init),
        )

        self.backbone.decoder = nn.Identity()
        self.backbone.decoder_lr = nn.Identity()
        self._convert_backbone_input_to_grayscale()

        dims_map = {
            "tiny": (96, 192, 384, 768),
            "base": (128, 256, 512, 1024),
            "large": (192, 384, 768, 1536),
        }
        name_l = str(cfg.model_name).lower()
        if "tiny" in name_l:
            in_ch = dims_map["tiny"]
        elif "large" in name_l:
            in_ch = dims_map["large"]
        else:
            in_ch = dims_map["base"]

        self.hidden = int(cfg.decoder_hidden_size)
        self.zero_init = bool(cfg.residual_zero_init)
        self.clamp_eval = bool(cfg.clamp_eval)
        self._head = HybridSRHead(
            in_ch,
            hidden=self.hidden,
            num_blocks=int(cfg.head_num_blocks),
            out_ch=1,
            window_size=int(cfg.head_window_size),
            attn_heads=int(cfg.head_attention_heads),
            zero_init=self.zero_init,
        )
        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def _replace_first_conv(self, module: nn.Module) -> bool:
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                new_conv = nn.Conv2d(
                    in_channels=1,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                    padding_mode=child.padding_mode,
                )
                with torch.no_grad():
                    new_conv.weight.copy_(child.weight.mean(dim=1, keepdim=True))
                    if child.bias is not None:
                        new_conv.bias.copy_(child.bias)
                setattr(module, name, new_conv)
                return True
            if self._replace_first_conv(child):
                return True
        return False

    def _convert_backbone_input_to_grayscale(self) -> None:
        replaced_hr = self._replace_first_conv(self.backbone.embeddings_hr)
        replaced_lr = self._replace_first_conv(self.backbone.embeddings_lr)
        if not (replaced_hr and replaced_lr):
            raise RuntimeError("Failed to convert CASWiT patch embeddings to 1-channel input.")

    def _adapt_state_dict_input_channels(self, state: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        adapted = {}
        for key, value in state.items():
            target = target_state.get(key)
            if (
                target is not None
                and torch.is_tensor(value)
                and torch.is_tensor(target)
                and value.ndim == 4
                and target.ndim == 4
                and value.shape[1] == 3
                and target.shape[1] == 1
                and value.shape[0] == target.shape[0]
                and value.shape[2:] == target.shape[2:]
            ):
                adapted[key] = value.mean(dim=1, keepdim=True)
            else:
                adapted[key] = value
        return adapted

    def load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and state.get("ema_model") is not None:
            state = state["ema_model"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        filtered = {k: v for k, v in state.items() if not (k.startswith("decoder.") or k.startswith("decoder_lr."))}
        filtered = self._adapt_state_dict_input_channels(filtered, self.backbone.state_dict())
        missing, unexpected = self.backbone.load_state_dict(filtered, strict=False)
        self._pretrained_missing = missing
        self._pretrained_unexpected = unexpected

    @staticmethod
    def _denorm01(x_n: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
        return x_n * std + mean

    @staticmethod
    def _norm(x01: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
        return (x01 - mean) / std

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == self.input_channels:
            return x
        if x.shape[1] == 3 and self.input_channels == 1:
            return x.mean(dim=1, keepdim=True)
        if x.shape[1] == 1 and self.input_channels == 3:
            return x.repeat(1, 3, 1, 1)
        raise ValueError(f"Unsupported input shape {tuple(x.shape)} for input_channels={self.input_channels}")

    def forward(self, x_hr_n: torch.Tensor, x_lr_n: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Inputs:
          x_hr_n: [B,3,H,W] normalized, bicubic base crop (channels identical)
          x_lr_n: [B,3,H,W] normalized, context-compressed crop (channels identical)
        Outputs:
          pred_n: [B,1,H,W] normalized prediction
          residual_n: [B,1,H,W] normalized residual
        """
        x_hr_n = self._prepare_input(x_hr_n)
        x_lr_n = self._prepare_input(x_lr_n)
        B, C, H, W = x_hr_n.shape

        import math

        x_hr_seq, _ = self.backbone.embeddings_hr(x_hr_n)
        x_lr_seq, _ = self.backbone.embeddings_lr(x_lr_n)

        N_hr = x_hr_seq.shape[1]
        N_lr = x_lr_seq.shape[1]
        H_hr = W_hr = int(math.sqrt(N_hr))
        H_lr = W_lr = int(math.sqrt(N_lr))
        dims_hr = (H_hr, W_hr)
        dims_lr = (H_lr, W_lr)

        features_hr = {}

        for idx, (stage_hr, stage_lr, ca) in enumerate(
            zip(self.backbone.encoder_layers_hr, self.backbone.encoder_layers_lr, self.backbone.cross_attn_blocks)
        ):
            for block in stage_hr.blocks:
                x_hr_seq = block(x_hr_seq, dims_hr)
                if isinstance(x_hr_seq, tuple):
                    x_hr_seq = x_hr_seq[0]

            for block in stage_lr.blocks:
                x_lr_seq = block(x_lr_seq, dims_lr)
                if isinstance(x_lr_seq, tuple):
                    x_lr_seq = x_lr_seq[0]

            x_hr_seq = self.backbone.hidden_states_norms_hr[f"stage{idx+1}"](x_hr_seq)
            x_lr_seq = self.backbone.hidden_states_norms_lr[f"stage{idx+1}"](x_lr_seq)

            Hh, Wh = dims_hr
            Hl, Wl = dims_lr
            Chr = x_hr_seq.shape[-1]
            Clr = x_lr_seq.shape[-1]

            feat_hr = x_hr_seq.transpose(1, 2).contiguous().view(B, Chr, Hh, Wh)
            feat_lr = x_lr_seq.transpose(1, 2).contiguous().view(B, Clr, Hl, Wl)

            fused_hr = ca(feat_hr, feat_lr)
            fused_hr_seq = fused_hr.flatten(2).transpose(1, 2).contiguous()

            if stage_hr.downsample is not None:
                fused_hr_seq = stage_hr.downsample(fused_hr_seq, dims_hr)
                dims_hr = (dims_hr[0] // 2, dims_hr[1] // 2)

            if stage_lr.downsample is not None:
                x_lr_seq = stage_lr.downsample(x_lr_seq, dims_lr)
                dims_lr = (dims_lr[0] // 2, dims_lr[1] // 2)

            features_hr[f"stage{idx+1}"] = fused_hr
            x_hr_seq = fused_hr_seq

        feats = (features_hr["stage1"], features_hr["stage2"], features_hr["stage3"], features_hr["stage4"])
        H_out, W_out = H // 2, W // 2
        residual_n = self._head(feats, out_hw=(H_out, W_out))  # [B,1,H/2,W/2] (x4 output)

        # Base at x4 resolution: downsample the x8 bicubic input by 2 (bicubic preserves linearity reasonably)
        base_1 = F.interpolate(x_hr_n[:, 0:1, :, :], size=(H_out, W_out), mode="bicubic", align_corners=False)
        pred_n = base_1 + residual_n

        if (not self.training) and self.clamp_eval:
            pred01 = self._denorm01(pred_n)
            pred01 = pred01.clamp(0.0, 1.0)
            pred_n = self._norm(pred01)

        return {"pred": pred_n, "residual": residual_n}
