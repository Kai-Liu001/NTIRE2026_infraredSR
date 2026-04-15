import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import MambaIRv2


def _parse_int_tuple(value):
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(value)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "params_ema", "params", "state_dict"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")


def _build_model_from_checkpoint(ckpt, device):
    cfg = ckpt.get("cfg", {})
    model = MambaIRv2(
        upscale=int(cfg.get("scale", 4)),
        img_size=64,
        in_chans=1,
        embed_dim=int(cfg.get("mamba_embed_dim", 174)),
        d_state=int(cfg.get("mamba_d_state", 16)),
        depths=_parse_int_tuple(cfg.get("mamba_depths", "6,6,6,6,6,6")),
        num_heads=_parse_int_tuple(cfg.get("mamba_num_heads", "6,6,6,6,6,6")),
        window_size=int(cfg.get("mamba_window_size", 16)),
        inner_rank=int(cfg.get("mamba_inner_rank", 64)),
        num_tokens=int(cfg.get("mamba_num_tokens", 128)),
        convffn_kernel_size=int(cfg.get("mamba_convffn_kernel_size", 5)),
        mlp_ratio=float(cfg.get("mamba_mlp_ratio", 2.0)),
        upsampler=str(cfg.get("mamba_upsampler", "pixelshuffle")),
        img_range=1.0,
    ).to(device)
    model.load_state_dict(_extract_state_dict(ckpt), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _load_input(path, device):
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    arr = np.array(img, dtype=np.uint8)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float() / 255.0
    return tensor.to(device)


def _save_output(sr_tensor, path):
    arr = sr_tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    rgb = np.repeat(arr[:, :, None], 3, axis=2)
    Image.fromarray(rgb, mode="RGB").save(path, compress_level=0)


def _is_image_file(path):
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}


@torch.inference_mode()
def main(model_dir, input_path, output_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_dir, map_location=device)
    model = _build_model_from_checkpoint(ckpt, device)

    input_dir = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([path for path in input_dir.iterdir() if path.is_file() and _is_image_file(path)])
    for image_path in image_paths:
        lr = _load_input(image_path, device)
        sr = model(lr).float().clamp(0.0, 1.0)
        _save_output(sr, output_dir / image_path.name)
