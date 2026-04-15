#!/usr/bin/env python3
"""
infer_sr_ir_context_x8_caswit_sliding_zip_tta_ensemble_shifted_ddp.py

Distributed multi-GPU inference + submission ZIP creator for Codabench.

Features:
- TTA: x1 / x4 / x8
- multi-checkpoint ensemble via --ckpts
- shifted sliding-window inference via --shifted_passes {1,2,4}
- distributed multi-GPU inference with torchrun

Usage on 4 GPUs:
torchrun --nproc_per_node=4 infer_sr_ir_context_x8_caswit_sliding_zip_tta_ensemble_shifted_ddp.py \
  --config config_sr_ir_context_x8_caswit_bicubic_residual.yaml \
  --ckpt best.pt \
  --lr_dir /path/to/test_lr \
  --out_dir /path/to/out \
  --zip_path /path/to/submission.zip \
  --device cuda \
  --overlap 64 \
  --tta 4 \
  --shifted_passes 2
"""

from __future__ import annotations

import argparse
import os
import time
import zipfile
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from PIL import Image

from model_sr_ir_context_x8_caswit_bicubic_residual import CASWiT_SR_IR_X8_Residual, SRModelCfg
from dataset_sr_ir_context_x8_black_residual_masked import SRIRX8DataCfg


# =========================
# Distributed helpers
# =========================

def ddp_is_available() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and ("LOCAL_RANK" in os.environ)


def ddp_init() -> None:
    if not ddp_is_available():
        return
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def ddp_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def ddp_world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main() -> bool:
    return ddp_rank() == 0


def ddp_barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# =========================
# IO / preprocessing
# =========================

def load_gray01(path: Path) -> torch.Tensor:
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).contiguous()  # [1,H,W]


def save_rgb_png_from_1ch01(path: Path, y01_1ch: torch.Tensor) -> None:
    y01_1ch = y01_1ch.detach().cpu().clamp(0, 1)
    arr = (y01_1ch.squeeze(0).numpy() * 255.0 + 0.5).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", compress_level=0)


def extract_with_black_pad(t: torch.Tensor, y0: int, x0: int, h: int, w: int) -> torch.Tensor:
    """Extract [C,h,w] from t [C,H,W] with black padding if outside."""
    C, H, W = t.shape
    out = torch.zeros((C, h, w), dtype=t.dtype, device=t.device)

    iy0 = max(0, y0)
    ix0 = max(0, x0)
    iy1 = min(H, y0 + h)
    ix1 = min(W, x0 + w)
    if iy1 <= iy0 or ix1 <= ix0:
        return out

    oy0 = iy0 - y0
    ox0 = ix0 - x0
    oy1 = oy0 + (iy1 - iy0)
    ox1 = ox0 + (ix1 - ix0)
    out[:, oy0:oy1, ox0:ox1] = t[:, iy0:iy1, ix0:ix1]
    return out


def normalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x - mean) / std


def denormalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return x * std + mean


def hann2d(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    wy = torch.hann_window(h, device=device, dtype=dtype, periodic=False)
    wx = torch.hann_window(w, device=device, dtype=dtype, periodic=False)
    win = (wy[:, None] * wx[None, :]).clamp_min(1e-6)
    return win.view(1, 1, h, w)


# =========================
# Model helpers
# =========================

def _model_forward(model: torch.nn.Module, x_hr: torch.Tensor, x_lr: torch.Tensor) -> Any:
    try:
        return model(x_hr=x_hr, x_lr=x_lr)
    except TypeError:
        pass
    try:
        return model(x_hr, x_lr)
    except TypeError:
        pass
    return model({"x_hr": x_hr, "x_lr": x_lr})


def _get_pred_tensor(out: Any) -> torch.Tensor:
    if isinstance(out, dict):
        if "pred" in out:
            return out["pred"]
        for k in ("pred_hr", "pred_ir", "y", "out"):
            if k in out:
                return out[k]
        for v in out.values():
            if torch.is_tensor(v):
                return v
        raise KeyError(f"Model output dict has no tensor entries: keys={list(out.keys())}")
    if torch.is_tensor(out):
        return out
    raise TypeError(f"Unsupported model output type: {type(out)}")


def _load_state_from_ckpt(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and ckpt.get("ema_model") is not None:
        return ckpt["ema_model"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _build_models(ckpt_paths: List[str], mcfg: SRModelCfg, device: torch.device) -> List[torch.nn.Module]:
    models: List[torch.nn.Module] = []
    for ckpt_path in ckpt_paths:
        model = CASWiT_SR_IR_X8_Residual(cfg=mcfg).to(device)
        state = _load_state_from_ckpt(ckpt_path)
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.eval()
        model._missing_keys = missing  # type: ignore[attr-defined]
        model._unexpected_keys = unexpected  # type: ignore[attr-defined]
        model._ckpt_path = ckpt_path  # type: ignore[attr-defined]
        models.append(model)
    return models


# =========================
# TTA helpers
# =========================

def _tta_modes(n: int) -> List[int]:
    if n == 1:
        return [0]
    if n == 4:
        return [0, 1, 2, 3]  # identity, hflip, vflip, hvflip
    if n == 8:
        return list(range(8))  # D4 group
    raise ValueError(f"Unsupported TTA count: {n}")


def _apply_tta(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0:
        return x
    if mode == 1:
        return torch.flip(x, dims=(-1,))
    if mode == 2:
        return torch.flip(x, dims=(-2,))
    if mode == 3:
        return torch.flip(x, dims=(-2, -1))
    if mode == 4:
        return x.transpose(-2, -1)
    if mode == 5:
        return torch.flip(x.transpose(-2, -1), dims=(-1,))
    if mode == 6:
        return torch.flip(x.transpose(-2, -1), dims=(-2,))
    if mode == 7:
        return torch.flip(x.transpose(-2, -1), dims=(-2, -1))
    raise ValueError(f"Unknown TTA mode: {mode}")


def _invert_tta(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0:
        return x
    if mode == 1:
        return torch.flip(x, dims=(-1,))
    if mode == 2:
        return torch.flip(x, dims=(-2,))
    if mode == 3:
        return torch.flip(x, dims=(-2, -1))
    if mode == 4:
        return x.transpose(-2, -1)
    if mode == 5:
        return torch.flip(x, dims=(-1,)).transpose(-2, -1)
    if mode == 6:
        return torch.flip(x, dims=(-2,)).transpose(-2, -1)
    if mode == 7:
        return torch.flip(x, dims=(-2, -1)).transpose(-2, -1)
    raise ValueError(f"Unknown TTA mode: {mode}")


# =========================
# Shifted tiling helpers
# =========================

def _make_positions(length: int, tile: int, stride: int, offset: int) -> List[int]:
    """
    Build tile start positions covering [0, length) with a given offset.
    Always includes the last valid position (length - tile).
    """
    if length <= tile:
        return [0]

    last = length - tile
    offset = max(0, min(offset, last))

    pos = list(range(offset, last + 1, stride))
    if 0 not in pos:
        pos = [0] + pos
    if pos[-1] != last:
        pos.append(last)

    return sorted(set(pos))


def _shift_offsets(stride: int, shifted_passes: int) -> List[Tuple[int, int]]:
    if shifted_passes == 1:
        return [(0, 0)]

    half = max(1, stride // 2)

    if shifted_passes == 2:
        return [(0, 0), (half, half)]

    if shifted_passes == 4:
        return [(0, 0), (half, 0), (0, half), (half, half)]

    raise ValueError("shifted_passes must be one of {1,2,4}")


# =========================
# Core inference
# =========================

@torch.no_grad()
def infer_one_image(
    model: torch.nn.Module,
    lr_1ch01: torch.Tensor,  # [1,h,w]
    *,
    scale: int,
    crop: int,
    ctx: int,
    mean: float,
    std: float,
    device: torch.device,
    overlap: int = 64,
    force_tiling: bool = False,
    shifted_passes: int = 1,
) -> torch.Tensor:
    """
    Returns: pred_1ch01 [1,H,W] in [0,1]
    """
    assert lr_1ch01.ndim == 3 and lr_1ch01.shape[0] == 1

    lr = lr_1ch01.to(device, non_blocking=True)

    # Bicubic upsample to x8 geometry
    lr_up = F.interpolate(lr.unsqueeze(0), scale_factor=scale, mode="bicubic", align_corners=False).squeeze(0)
    H8, W8 = int(lr_up.shape[1]), int(lr_up.shape[2])

    # Output is on x4 grid
    H, W = H8 // 2, W8 // 2
    crop_out = crop // 2

    # Small image path
    if (not force_tiling) and (H <= crop_out and W <= crop_out):
        pad_h = max(0, crop - H8)
        pad_w = max(0, crop - W8)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        x_hr_1 = lr_up
        if pad_h or pad_w:
            x_hr_1 = F.pad(x_hr_1, (left, right, top, bottom), mode="constant", value=0.0)

        cy = crop // 2
        cx = crop // 2
        ctx_y0 = cy - ctx // 2
        ctx_x0 = cx - ctx // 2
        ctx_patch_1 = extract_with_black_pad(x_hr_1, ctx_y0, ctx_x0, ctx, ctx)
        x_lr_1 = F.interpolate(ctx_patch_1.unsqueeze(0), size=(crop, crop), mode="bicubic", align_corners=False).squeeze(0)

        x_hr_n = normalize(x_hr_1, mean, std).unsqueeze(0)
        x_lr_n = normalize(x_lr_1, mean, std).unsqueeze(0)

        out = _model_forward(model, x_hr_n, x_lr_n)
        pred_n = _get_pred_tensor(out)
        pred01 = denormalize(pred_n, mean, std).clamp(0, 1)

        if pad_h or pad_w:
            top4 = top // 2
            left4 = left // 2
            pred01 = pred01[:, :, top4:top4 + H, left4:left4 + W]
        return pred01.squeeze(0)

    # Tiled inference
    overlap_out = max(0, overlap // 2)
    stride = max(1, crop_out - overlap_out)

    acc_global = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
    wsum_global = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
    win = hann2d(crop_out, crop_out, device=device, dtype=torch.float32)

    offsets = _shift_offsets(stride, shifted_passes)

    for off_y, off_x in offsets:
        ys = _make_positions(H, crop_out, stride, off_y)
        xs = _make_positions(W, crop_out, stride, off_x)

        for y0 in ys:
            for x0 in xs:
                y0_8 = y0 * 2
                x0_8 = x0 * 2
                x_hr_1 = extract_with_black_pad(lr_up, y0_8, x0_8, crop, crop)

                cy4 = y0 + crop_out // 2
                cx4 = x0 + crop_out // 2
                cy8 = cy4 * 2
                cx8 = cx4 * 2
                ctx_y0 = cy8 - ctx // 2
                ctx_x0 = cx8 - ctx // 2
                ctx_patch_1 = extract_with_black_pad(lr_up, ctx_y0, ctx_x0, ctx, ctx)
                x_lr_1 = F.interpolate(
                    ctx_patch_1.unsqueeze(0),
                    size=(crop, crop),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)

                x_hr_n = normalize(x_hr_1, mean, std).unsqueeze(0)
                x_lr_n = normalize(x_lr_1, mean, std).unsqueeze(0)

                out = _model_forward(model, x_hr_n, x_lr_n)
                pred_n = _get_pred_tensor(out)
                pred01 = denormalize(pred_n, mean, std).clamp(0, 1)

                acc_global[:, :, y0:y0 + crop_out, x0:x0 + crop_out] += pred01 * win
                wsum_global[:, :, y0:y0 + crop_out, x0:x0 + crop_out] += win

    pred01_full = acc_global / wsum_global.clamp_min(1e-6)
    return pred01_full.squeeze(0)


@torch.no_grad()
def infer_one_image_tta_ensemble(
    models: List[torch.nn.Module],
    lr_1ch01: torch.Tensor,
    *,
    scale: int,
    crop: int,
    ctx: int,
    mean: float,
    std: float,
    device: torch.device,
    overlap: int,
    force_tiling: bool,
    tta: int,
    shifted_passes: int,
) -> torch.Tensor:
    tta_modes = _tta_modes(tta)

    pred_models: List[torch.Tensor] = []
    for model in models:
        pred_tta: List[torch.Tensor] = []

        for mode in tta_modes:
            lr_aug = _apply_tta(lr_1ch01, mode).contiguous()

            pred_aug = infer_one_image(
                model,
                lr_aug,
                scale=scale,
                crop=crop,
                ctx=ctx,
                mean=mean,
                std=std,
                device=device,
                overlap=overlap,
                force_tiling=force_tiling,
                shifted_passes=shifted_passes,
            )

            pred = _invert_tta(pred_aug, mode).contiguous()
            pred_tta.append(pred)

        pred_model = torch.stack(pred_tta, dim=0).mean(dim=0)
        pred_models.append(pred_model)

    return torch.stack(pred_models, dim=0).mean(dim=0)


# =========================
# Readme / stats
# =========================

def build_readme(runtime_s: float, use_gpu: bool, extra_data: bool, other: str) -> str:
    return (
        f"runtime per image [s] : {runtime_s:.4f}\n"
        f"CPU[1] / GPU[0] : {0 if use_gpu else 1}\n"
        f"Extra Data [1] / No Extra Data [0] : {1 if extra_data else 0}\n"
        f"Other description : {other}\n"
    )


def gather_runtime_stats(local_sum: float, local_count: int) -> Tuple[float, int]:
    if not dist.is_initialized():
        return local_sum, local_count

    payload = [None for _ in range(ddp_world())]
    dist.all_gather_object(payload, (float(local_sum), int(local_count)))

    total_sum = sum(x[0] for x in payload)
    total_count = sum(x[1] for x in payload)
    return total_sum, total_count


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="YAML config used for training.")
    ap.add_argument("--ckpt", type=str, default=None, help="Single checkpoint (.pt).")
    ap.add_argument("--ckpts", nargs="+", type=str, default=None, help="Multiple checkpoints for ensemble.")
    ap.add_argument("--lr_dir", required=True, type=str, help="Folder with LR test images (PNG).")
    ap.add_argument("--out_dir", required=True, type=str, help="Output folder for HR PNGs.")
    ap.add_argument("--zip_path", required=True, type=str, help="Output ZIP path.")
    ap.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    ap.add_argument("--overlap", default=64, type=int, help="Tile overlap in HR pixels.")
    ap.add_argument("--force_tiling", action="store_true", help="Always use tiling even for small images.")
    ap.add_argument("--tta", default=1, type=int, choices=[1, 4, 8], help="TTA count.")
    ap.add_argument("--shifted_passes", default=1, type=int, choices=[1, 2, 4], help="Shifted tiling passes.")
    ap.add_argument("--extra_data", action="store_true", help="Set Extra Data=1 in readme.")
    ap.add_argument(
        "--other_desc",
        default="CASWiT backbone + residual SR head + TTA/ensemble/shifted-tiling + distributed inference.",
        type=str,
    )
    args = ap.parse_args()

    if args.ckpts is None and args.ckpt is None:
        raise ValueError("Provide either --ckpt or --ckpts.")
    if args.ckpts is not None and args.ckpt is not None:
        raise ValueError("Use either --ckpt or --ckpts, not both.")

    ckpt_paths = args.ckpts if args.ckpts is not None else [args.ckpt]
    if len(ckpt_paths) == 0:
        raise ValueError("No checkpoint path provided.")

    ddp_init()
    rank = ddp_rank()
    world = ddp_world()

    cfg = yaml.safe_load(open(args.config, "r"))

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

    if args.device == "cuda" and torch.cuda.is_available():
        if ddp_is_available():
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_gpu = device.type == "cuda"

    models = _build_models(ckpt_paths, mcfg, device)

    lr_dir = Path(args.lr_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lr_files = sorted([p for p in lr_dir.glob("*.png")])
    if len(lr_files) == 0:
        raise FileNotFoundError(f"No .png found in {lr_dir}")

    # Partition images across ranks
    lr_files_local = lr_files[rank::world]

    if is_main():
        print(f"Total images: {len(lr_files)}")
        print(f"World size: {world}")
        print(f"TTA: x{args.tta}")
        print(f"Ensemble checkpoints: {len(ckpt_paths)}")
        print(f"Shifted passes: {args.shifted_passes}")
        print(f"Images per rank: {[len(lr_files[r::world]) for r in range(world)]}")

    local_runtime_sum = 0.0
    local_runtime_count = 0

    for idx, p in enumerate(lr_files_local):
        lr_1ch01 = load_gray01(p)

        if use_gpu:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        pred_1ch01 = infer_one_image_tta_ensemble(
            models,
            lr_1ch01,
            scale=scale,
            crop=crop,
            ctx=ctx,
            mean=mean,
            std=std,
            device=device,
            overlap=int(args.overlap),
            force_tiling=bool(args.force_tiling),
            tta=int(args.tta),
            shifted_passes=int(args.shifted_passes),
        )

        if use_gpu:
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        local_runtime_sum += (t1 - t0)
        local_runtime_count += 1

        save_rgb_png_from_1ch01(out_dir / p.name, pred_1ch01)

        print(
            f"[rank {rank}] {idx + 1}/{len(lr_files_local)} done: {p.name} "
            f"({t1 - t0:.4f}s)"
        )

    # Wait until all ranks finished writing PNGs
    ddp_barrier()

    total_runtime_sum, total_runtime_count = gather_runtime_stats(local_runtime_sum, local_runtime_count)
    runtime_avg = total_runtime_sum / max(1, total_runtime_count)

    if is_main():
        ckpt_names = ", ".join(Path(p).name for p in ckpt_paths)
        other_desc = (
            f"{args.other_desc} | "
            f"tta={args.tta} | "
            f"ensemble={len(ckpt_paths)} | "
            f"shifted_passes={args.shifted_passes} | "
            f"world_size={world} | "
            f"ckpts={ckpt_names}"
        )

        readme = build_readme(runtime_avg, use_gpu=use_gpu, extra_data=bool(args.extra_data), other=other_desc)
        readme_path = out_dir / "readme.txt"
        readme_path.write_text(readme, encoding="utf-8")

        zip_path = Path(args.zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for p in sorted(out_dir.glob("*.png")):
                zf.write(p, arcname=p.name)
            zf.write(readme_path, arcname="readme.txt")

        print(f"Saved {len(lr_files)} images to: {out_dir}")
        print(f"Submission ZIP: {zip_path}")
        print(f"Avg runtime per image [s]: {runtime_avg:.4f}")

        for i, model in enumerate(models):
            missing = getattr(model, "_missing_keys", [])
            unexpected = getattr(model, "_unexpected_keys", [])
            ckpt_path = getattr(model, "_ckpt_path", f"ckpt_{i}")
            print(f"[{i}] {ckpt_path}")
            if missing:
                print(f"[WARN][{i}] Missing keys (first 10): {missing[:10]}{'...' if len(missing) > 10 else ''}")
            if unexpected:
                print(f"[WARN][{i}] Unexpected keys (first 10): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    ddp_barrier()
    ddp_cleanup()


if __name__ == "__main__":
    main()
