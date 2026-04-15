"""
CASWiT: Context-Aware Swin Transformer for Ultra-High Resolution Semantic Segmentation

This module implements the main CASWiT model architecture with dual-branch
high-resolution and low-resolution processing with cross-attention fusion.
"""

import math
import copy
from typing import Dict
import torch
import torch.nn as nn
from transformers import SegformerConfig, SwinConfig, UperNetConfig, UperNetForSemanticSegmentation
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()



class DropPath(nn.Module):
    """Drop path (stochastic depth) regularization module."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
    
    def forward(self, x):
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class CrossFusionBlock(nn.Module):
    """
    Cross-attention fusion block that enables HR features to attend to LR features.
    
    Implements pre-norm cross-attention (Q=HR, K/V=LR).
    
    Args:
        C_hr: Channel dimension of HR features
        C_lr: Channel dimension of LR features
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        drop: Dropout rate
        drop_path: Drop path rate
    """
    def __init__(self, C_hr: int, C_lr: int, num_heads: int = 8,
                 mlp_ratio: float = 4.0, drop: float = 0.0, drop_path: float = 0.1):
        super().__init__()

        self.norm_q = nn.LayerNorm(C_hr)
        self.norm_kv = nn.LayerNorm(C_lr)
        self.attn = nn.MultiheadAttention(
            embed_dim=C_hr, num_heads=num_heads, kdim=C_lr, vdim=C_lr,
            dropout=drop, batch_first=True
        )

        hidden = int(C_hr * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(C_hr),
            nn.Linear(C_hr, hidden),
            nn.GELU(),
            nn.Linear(hidden, C_hr),
        )

    def forward(self, x_hr: torch.Tensor, x_lr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through cross-attention fusion block.
        
        Args:
            x_hr: HR features [B, C_hr, H_hr, W_hr]
            x_lr: LR features [B, C_lr, H_lr, W_lr]
            
        Returns:
            Fused HR features [B, C_hr, H_hr, W_hr]
        """
        B, C_hr, H_hr, W_hr = x_hr.shape
        _, C_lr, H_lr, W_lr = x_lr.shape

        # Flatten to sequences
        q  = x_hr.flatten(2).transpose(1, 2)    # [B, N_hr, C_hr]
        kv = x_lr.flatten(2).transpose(1, 2)    # [B, N_lr, C_lr]

        # Pre-norm
        qn  = self.norm_q(q)
        kvn = self.norm_kv(kv)

        attn_out, _ = self.attn(qn, kvn, kvn)  # [B, N_hr, C_hr]

        # Residual connection + MLP
        y = q + attn_out
        y = y + self.mlp(y)
        
        return y.transpose(1, 2).view(B, C_hr, H_hr, W_hr)


class CASWiT(nn.Module):
    """
    CASWiT: Context-Aware Swin Transformer for Ultra-High Resolution Semantic Segmentation.
    
    Dual-branch architecture with:
    - HR branch: Processes high-resolution crops
    - LR branch: Processes low-resolution context
    - Cross-attention fusion at each encoder stage
    
    Args:
        num_head_xa: Number of cross-attention heads
        num_classes: Number of segmentation classes
        model_name: HuggingFace model identifier for UPerNet-Swin
        mlp_ratio: MLP expansion ratio in fusion blocks
        drop_path: Drop path rate
    """
    def __init__(self, num_head_xa: int = 1, num_classes: int = 12, 
                 model_name: str = "openmmlab/upernet-swin-tiny", 
                 mlp_ratio: float = 4.0, drop_path: float = 0.1,
                 backbone_init: str = "hf_pretrained"):
        super().__init__()
        backbone_init = str(backbone_init).lower()
        if backbone_init not in {"hf_pretrained", "scratch"}:
            raise ValueError(f"Unsupported backbone_init={backbone_init}")

        def _infer_swin_variant(name: str) -> Dict[str, object]:
            name_l = name.lower()
            if "tiny" in name_l:
                return {
                    "embed_dim": 96,
                    "depths": [2, 2, 6, 2],
                    "num_heads": [3, 6, 12, 24],
                    "window_size": 7,
                }
            if "large" in name_l:
                return {
                    "embed_dim": 192,
                    "depths": [2, 2, 18, 2],
                    "num_heads": [6, 12, 24, 48],
                    "window_size": 12,
                }
            return {
                "embed_dim": 128,
                "depths": [2, 2, 18, 2],
                "num_heads": [4, 8, 16, 32],
                "window_size": 12,
            }

        def _build_scratch_upernet() -> UperNetForSemanticSegmentation:
            swin_kwargs = _infer_swin_variant(model_name)
            window_size = int(swin_kwargs["window_size"])
            swin_kwargs = {k: v for k, v in swin_kwargs.items() if k != "window_size"}
            backbone_cfg = SwinConfig(
                image_size=512,
                patch_size=4,
                num_channels=3,
                window_size=window_size,
                mlp_ratio=4.0,
                qkv_bias=True,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                drop_path_rate=drop_path,
                use_absolute_embeddings=False,
                encoder_stride=32,
                out_features=["stage1", "stage2", "stage3", "stage4"],
                **swin_kwargs,
            )
            cfg = UperNetConfig(
                backbone_config=backbone_cfg,
                use_pretrained_backbone=False,
                num_labels=num_classes,
                hidden_size=512,
                auxiliary_in_channels=int(swin_kwargs["embed_dim"]) * 4,
            )
            return UperNetForSemanticSegmentation(cfg)

        def _load_hf_upernet() -> UperNetForSemanticSegmentation:
            pretrained_kwargs = dict(
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
            return UperNetForSemanticSegmentation.from_pretrained(
                model_name,
                **pretrained_kwargs,
            )

        # Load two UPerNet backbones (HR and LR branches)
        if backbone_init == "scratch":
            model_hr = _build_scratch_upernet()
            model_lr = _build_scratch_upernet()
        else:
            model_hr = _load_hf_upernet()
            model_lr = copy.deepcopy(model_hr)
        
        # Extract HR branch components
        self.embeddings_hr = model_hr.backbone.embeddings
        self.encoder_layers_hr = model_hr.backbone.encoder.layers
        self.hidden_states_norms_hr = model_hr.backbone.hidden_states_norms
        self.decoder = None  # placeholder, set after dims inference

        # Extract LR branch components
        self.embeddings_lr = model_lr.backbone.embeddings
        self.encoder_layers_lr = model_lr.backbone.encoder.layers
        self.hidden_states_norms_lr = model_lr.backbone.hidden_states_norms
        self.decoder_lr = None  # placeholder, set after dims inference

        # Cross-attention blocks at each stage
        # Dimensions: tiny:[96, 192, 384, 768] base:[128, 256, 512, 1024] large:[192, 384, 768, 1536]
        dims_map = {
            "tiny": [96, 192, 384, 768],
            "base": [128, 256, 512, 1024],
            "large": [192, 384, 768, 1536]
        }
        # Infer dimensions from model name
        if "tiny" in model_name.lower():
            dims = dims_map["tiny"]
        elif "large" in model_name.lower():
            dims = dims_map["large"]
        else:
            dims = dims_map["base"]  # default to base
        
        segformer_cfg = SegformerConfig(
            num_labels=num_classes,
            hidden_sizes=dims,
            num_encoder_blocks=4,
            decoder_hidden_size=512,
            classifier_dropout_prob=0.0,
        )
        self.decoder = SegformerDecodeHead(segformer_cfg)
        self.decoder_lr = SegformerDecodeHead(segformer_cfg)

        self.cross_attn_blocks = nn.ModuleList([
            CrossFusionBlock(dim, dim, num_heads=num_head_xa,
                           mlp_ratio=mlp_ratio, drop=0.0, drop_path=drop_path) 
            for dim in dims
        ])

    def forward(self, x_hr: torch.Tensor, x_lr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CASWiT model.
        
        Args:
            x_hr: HR input images [B, 3, H_hr, W_hr]
            x_lr: LR input images [B, 3, H_lr, W_lr]
            
        Returns:
            Dictionary with 'logits_hr' and 'logits_lr' segmentation logits
        """
        B = x_hr.size(0)
        
        # Patch embeddings
        x_hr_seq, _ = self.embeddings_hr(x_hr)
        x_lr_seq, _ = self.embeddings_lr(x_lr)

        N_hr, C_hr = x_hr_seq.shape[1], x_hr_seq.shape[2]
        N_lr, C_lr = x_lr_seq.shape[1], x_lr_seq.shape[2]
        H_hr = W_hr = int(math.sqrt(N_hr))
        H_lr = W_lr = int(math.sqrt(N_lr))
        dims_hr = (H_hr, W_hr)
        dims_lr = (H_lr, W_lr)

        features_hr: Dict[str, torch.Tensor] = {}
        features_lr: Dict[str, torch.Tensor] = {}

        # Process through encoder stages with cross-attention fusion
        for idx, (stage_hr, stage_lr, ca) in enumerate(zip(
            self.encoder_layers_hr, self.encoder_layers_lr, self.cross_attn_blocks
        )):
            # HR branch blocks
            for block in stage_hr.blocks:
                x_hr_seq = block(x_hr_seq, dims_hr)
                if isinstance(x_hr_seq, tuple):
                    x_hr_seq = x_hr_seq[0]
            
            # LR branch blocks
            for block in stage_lr.blocks:
                x_lr_seq = block(x_lr_seq, dims_lr)
                if isinstance(x_lr_seq, tuple):
                    x_lr_seq = x_lr_seq[0]

            # Layer normalization
            x_hr_seq = self.hidden_states_norms_hr[f"stage{idx+1}"](x_hr_seq)
            x_lr_seq = self.hidden_states_norms_lr[f"stage{idx+1}"](x_lr_seq)

            H_hr, W_hr = dims_hr
            H_lr, W_lr = dims_lr
            C_hr = x_hr_seq.shape[-1]
            C_lr = x_lr_seq.shape[-1]

            # Reshape to spatial format
            feat_hr = x_hr_seq.transpose(1, 2).contiguous().view(B, C_hr, H_hr, W_hr)
            feat_lr = x_lr_seq.transpose(1, 2).contiguous().view(B, C_lr, H_lr, W_lr)

            fused_hr = ca(feat_hr, feat_lr)
            fused_hr_seq = fused_hr.flatten(2).transpose(1, 2).contiguous()

            # Downsample if stage has it
            if stage_hr.downsample is not None:
                fused_hr_seq = stage_hr.downsample(fused_hr_seq, dims_hr)
                dims_hr = (dims_hr[0] // 2, dims_hr[1] // 2)
            if stage_lr.downsample is not None:
                x_lr_seq = stage_lr.downsample(x_lr_seq, dims_lr)
                dims_lr = (dims_lr[0] // 2, dims_lr[1] // 2)

            features_hr[f"stage{idx+1}"] = fused_hr
            features_lr[f"stage{idx+1}"] = feat_lr
            x_hr_seq = fused_hr_seq

        # Decode HR features
        features_tuple = (
            features_hr["stage1"],
            features_hr["stage2"],
            features_hr["stage3"],
            features_hr["stage4"],
        )
        logits = self.decoder(features_tuple)
        
        # Decode LR features (for auxiliary supervision)
        features_tuple_lr = (
            features_lr["stage1"],
            features_lr["stage2"],
            features_lr["stage3"],
            features_lr["stage4"],
        )
        logits_lr = self.decoder_lr(features_tuple_lr)
        
        return {"logits_hr": logits, "logits_lr": logits_lr}



def _test_segformer_head():
    """Quick, offline test for SegFormer head input/output shapes."""
    # Example dims for Swin-Tiny stages:
    dims = [96, 192, 384, 768]
    cfg = SegformerConfig(
        num_labels=7,
        hidden_sizes=dims,
        num_encoder_blocks=4,
        decoder_hidden_size=512,
        classifier_dropout_prob=0.0,
    )
    head = SegformerDecodeHead(cfg)

    B = 2
    # Stage resolutions typically differ by /2 each time; here we mimic that.
    f1 = torch.randn(B, dims[0], 128, 128)
    f2 = torch.randn(B, dims[1], 64, 64)
    f3 = torch.randn(B, dims[2], 32, 32)
    f4 = torch.randn(B, dims[3], 16, 16)

    logits = head((f1, f2, f3, f4))
    assert logits.shape == (B, cfg.num_labels, 128, 128), f"Unexpected logits shape: {logits.shape}"
    return logits.shape


if __name__ == "__main__":
    print("SegFormer head test logits shape:", _test_segformer_head())
