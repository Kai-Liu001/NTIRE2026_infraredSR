from __future__ import annotations

import os
import sys
from dataclasses import fields
from pathlib import Path

import torch
import yaml

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from infer_sr_ir_context_x8_caswit_sliding_zip_tta_ensemble_shifted_windows import (
    _build_models,
    infer_one_image_tta_ensemble,
    load_gray01,
    save_rgb_png_from_1ch01,
)
from model_sr_ir_context_x8_caswit_bicubic_residual import SRModelCfg
from dataset_sr_ir_context_x8_black_residual_masked import SRIRX8DataCfg


def _resolve_lr_dir(input_path: str | os.PathLike[str]) -> Path:
    root = Path(input_path)
    candidates = [
        root / "LQ",
        root / "lq",
        root / "LR",
        root / "lr",
        root,
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.png")):
            return candidate
    raise FileNotFoundError(f"No LR PNG directory found under {root}")


def _resolve_ckpts(model_dir: str | os.PathLike[str]) -> list[Path]:
    model_dir = Path(model_dir)
    if model_dir.is_file():
        return [model_dir]

    candidates = []
    for candidate in (model_dir / "last.pt", model_dir / "best.pt"):
        if candidate.exists():
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    return candidates


def main(model_dir, input_path, output_path, device=None):
    root = Path(__file__).resolve().parent
    cfg_path = root / "config_sr_ir_context_x8_caswit_bicubic_residual_landsat_v2.yaml"

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = SRIRX8DataCfg(**cfg.get("data", {}))
    crop = int(data_cfg.hr_crop)
    ctx = int(data_cfg.ctx_crop)
    scale = int(data_cfg.scale_up)
    mean = float(data_cfg.normalize_mean)
    std = float(data_cfg.normalize_std)

    model_cfg_dict = cfg.get("model", {})
    allowed = {f.name for f in fields(SRModelCfg)}
    filtered_cfg = {k: v for k, v in model_cfg_dict.items() if k in allowed}
    mcfg = SRModelCfg(**filtered_cfg)

    if device is None:
        run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif hasattr(device, "type"):
        run_device = device
    else:
        run_device = torch.device(str(device))

    ckpt_paths = _resolve_ckpts(model_dir)
    models = _build_models([str(path) for path in ckpt_paths], mcfg, run_device)

    lr_dir = _resolve_lr_dir(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    tta = 4
    shifted_passes = 2
    overlap = 64
    force_tiling = False

    lr_files = sorted(lr_dir.glob("*.png"))
    if not lr_files:
        raise FileNotFoundError(f"No PNG images found in {lr_dir}")

    for lr_path in lr_files:
        lr_1ch01 = load_gray01(lr_path)
        pred_1ch01 = infer_one_image_tta_ensemble(
            models,
            lr_1ch01,
            scale=scale,
            crop=crop,
            ctx=ctx,
            mean=mean,
            std=std,
            device=run_device,
            overlap=overlap,
            force_tiling=force_tiling,
            tta=tta,
            shifted_passes=shifted_passes,
        )
        save_rgb_png_from_1ch01(output_dir / lr_path.name, pred_1ch01)
