"""
NTIRE 2026 Remote Sensing Infrared Image Super-Resolution
Team 07 — NTR (University of Illinois at Urbana-Champaign)

Model: TimeDiffiT_ResNet_color_128 with built-in PixelShuffle SR head.
Pretrained via MDAE, fine-tuned with L1+MSE+SWT+Synthetic SR loss.
Inference: 128x128 tiles, 32px overlap, 8-fold geometric self-ensemble.
"""

import os
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .TimeDiffiT_ResNet_color_128_arch import TimeDiffiT_ResNet_color_128


# ---------------------------------------------------------------------------
# Inference utilities
# ---------------------------------------------------------------------------

def _read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_tensor(img, device):
    arr = np.ascontiguousarray(img.transpose((2, 0, 1)))
    return torch.from_numpy(arr).float().unsqueeze(0).to(device) / 255.0


def _to_uint8(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _make_tiles(length, tile, overlap):
    if tile <= 0 or length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _run_model(model, x, scale):
    """Single forward pass with [-1,1] normalisation."""
    xn = x * 2.0 - 1.0
    t = torch.zeros(xn.shape[0], device=xn.device, dtype=xn.dtype)
    with torch.no_grad():
        out = model(x1=xn, time=t)
    return torch.clamp((out + 1.0) * 0.5, 0.0, 1.0)


def _run_tiled(model, lr, scale, tile, overlap):
    _, _, h, w = lr.shape
    if tile <= 0 or (h <= tile and w <= tile):
        return _run_model(model, lr, scale)

    out = torch.zeros((1, 3, h * scale, w * scale), device=lr.device, dtype=lr.dtype)
    wt = torch.zeros_like(out)
    for y in _make_tiles(h, tile, overlap):
        for x in _make_tiles(w, tile, overlap):
            patch = lr[:, :, y:y + tile, x:x + tile]
            p_out = _run_model(model, patch, scale)
            ph, pw = p_out.shape[2:]
            out[:, :, y * scale:y * scale + ph, x * scale:x * scale + pw] += p_out
            wt[:, :, y * scale:y * scale + ph, x * scale:x * scale + pw] += 1.0
    return out / wt.clamp_min(1.0)


def _apply_transform(x, t):
    if t >= 4:
        x = torch.flip(x, dims=[3])
    k = t % 4
    if k > 0:
        x = torch.rot90(x, k=k, dims=[2, 3])
    return x


def _invert_transform(x, t):
    k = t % 4
    if k > 0:
        x = torch.rot90(x, k=4 - k, dims=[2, 3])
    if t >= 4:
        x = torch.flip(x, dims=[3])
    return x


def _run_ensemble(model, lr, scale, tile, overlap):
    """8-fold geometric self-ensemble (dihedral group D4)."""
    accum = None
    for t in range(8):
        sr = _run_tiled(model, _apply_transform(lr, t), scale, tile, overlap)
        sr = _invert_transform(sr, t)
        accum = sr if accum is None else accum + sr
    return accum / 8.0


# ---------------------------------------------------------------------------
# Required interface: main(model_dir, input_path, output_path, device)
# ---------------------------------------------------------------------------

def main(model_dir, input_path, output_path, device=None):
    """
    Args:
        model_dir  : path to checkpoint (.pth)
        input_path : folder containing LR PNG images
        output_path: folder to write SR PNG images
        device     : torch.device (cuda or cpu)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)

    # --- build model (matches training config) ---
    model = TimeDiffiT_ResNet_color_128(
        dim=128,
        init_dim=128,
        out_dim=3,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
    ).to(device)

    state = torch.load(model_dir, map_location=device, weights_only=False)
    clean = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(clean, strict=False)
    model.task = "sr"  # ensure SR head is used (not denoise path)
    model.sr_scale = 4
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    TILE = 128
    OVERLAP = 32
    SCALE = 4

    os.makedirs(output_path, exist_ok=True)
    img_paths = sorted([
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ])

    for i, path in enumerate(img_paths):
        name = os.path.basename(path)
        lr = _to_tensor(_read_rgb(path), device)
        sr = _run_ensemble(model, lr, SCALE, TILE, OVERLAP)
        sr_uint8 = _to_uint8(sr)
        cv2.imwrite(os.path.join(output_path, name), cv2.cvtColor(sr_uint8, cv2.COLOR_RGB2BGR))
        logger.info("[%d/%d] %s", i + 1, len(img_paths), name)
