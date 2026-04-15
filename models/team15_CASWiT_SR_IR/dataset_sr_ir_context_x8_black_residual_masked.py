# dataset_sr_ir_context_x8_black_residual_masked.py
# 1-band (grayscale) x8 SR dataset for CASWiT.
#
# Modified:
# - Added spatial augmentations during training only:
#   * random horizontal flip
#   * random vertical flip
#   * random rotation by k*90°
# - valid_bbox is recomputed after augmentation from valid_mask

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image


@dataclass
class SRIRX8DataCfg:
    hr_crop: int = 512
    ctx_crop: int = 1024
    scale_up: int = 4
    normalize_mean: float = 0.5
    normalize_std: float = 0.5
    lr_suffix: str = "x4"  # not used below unless you customize lr naming


def _to_chw_float01_gray(img: Image.Image) -> torch.Tensor:
    """PIL grayscale -> float tensor [1,H,W] in [0,1], preserving 16-bit inputs."""
    if img.mode in {"L", "I;16", "I;16L", "I;16B", "I"}:
        arr = np.array(img)
    else:
        arr = np.array(img.convert("L"))

    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        if arr.max(initial=0.0) > 1.0:
            arr /= max(1e-12, float(arr.max()))

    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def _normalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x - mean) / std


def _crop_hw(t: torch.Tensor, y: int, x: int, h: int, w: int) -> torch.Tensor:
    return t[:, y : y + h, x : x + w]


def _extract_with_black_pad(t: torch.Tensor, y0: int, x0: int, h: int, w: int) -> torch.Tensor:
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


def _pad_tensor_to_min_hw_black_with_meta(
    t: torch.Tensor, min_h: int, min_w: int
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """
    Black-pad a CHW tensor to at least (min_h, min_w), padding symmetrically.

    Returns:
      - padded tensor [C,H',W']
      - (top, left) padding offsets applied to original content
      - (H0, W0) original spatial size
    """
    assert t.ndim == 3, f"Expected [C,H,W], got {tuple(t.shape)}"
    _, H0, W0 = t.shape

    pad_h = max(0, min_h - H0)
    pad_w = max(0, min_w - W0)

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if pad_h == 0 and pad_w == 0:
        return t, (0, 0), (H0, W0)

    t_pad = F.pad(t, (left, right, top, bottom), mode="constant", value=0.0)
    return t_pad, (top, left), (H0, W0)


def _spatial_augment(
    x_hr_1: torch.Tensor,
    x_lr_1: torch.Tensor,
    y_1: torch.Tensor,
    x_base_1: torch.Tensor,
    valid_crop: torch.Tensor,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply only spatial augmentations, consistently across all tensors."""
    if rng.random() < 0.5:
        x_hr_1 = torch.flip(x_hr_1, dims=[2])
        x_lr_1 = torch.flip(x_lr_1, dims=[2])
        y_1 = torch.flip(y_1, dims=[2])
        x_base_1 = torch.flip(x_base_1, dims=[2])
        valid_crop = torch.flip(valid_crop, dims=[2])

    if rng.random() < 0.5:
        x_hr_1 = torch.flip(x_hr_1, dims=[1])
        x_lr_1 = torch.flip(x_lr_1, dims=[1])
        y_1 = torch.flip(y_1, dims=[1])
        x_base_1 = torch.flip(x_base_1, dims=[1])
        valid_crop = torch.flip(valid_crop, dims=[1])

    k = rng.randint(0, 3)
    if k > 0:
        x_hr_1 = torch.rot90(x_hr_1, k=k, dims=[1, 2])
        x_lr_1 = torch.rot90(x_lr_1, k=k, dims=[1, 2])
        y_1 = torch.rot90(y_1, k=k, dims=[1, 2])
        x_base_1 = torch.rot90(x_base_1, k=k, dims=[1, 2])
        valid_crop = torch.rot90(valid_crop, k=k, dims=[1, 2])

    return x_hr_1, x_lr_1, y_1, x_base_1, valid_crop


def _bbox_from_valid_mask(valid_crop: torch.Tensor) -> torch.Tensor:
    """Compute (vy0, vy1, vx0, vx1) from valid_mask [1,H,W]."""
    ys, xs = torch.where(valid_crop[0] > 0.5)
    if ys.numel() == 0:
        vy0 = vy1 = vx0 = vx1 = 0
    else:
        vy0 = int(ys.min().item())
        vy1 = int(ys.max().item()) + 1
        vx0 = int(xs.min().item())
        vx1 = int(xs.max().item()) + 1
    return torch.tensor([vy0, vy1, vx0, vx1], dtype=torch.int64)


class SRIRX8Dataset(Dataset):
    """Grayscale SR dataset producing 1-channel inputs."""

    def __init__(
        self,
        hr_dir: str | Path,
        lr_dir: str | Path,
        cfg: SRIRX8DataCfg,
        split: str = "train",
        seed: int = 1337,
    ):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.cfg = cfg
        self.split = split
        self.rng = random.Random(seed)

        self.hr_files: List[Path] = sorted([p for p in self.hr_dir.glob("*.png")])
        if len(self.hr_files) == 0:
            raise FileNotFoundError(f"No HR .png found in {self.hr_dir}")

        self.lr_files: List[Path] = []
        for p in self.hr_files:
            lr_name = p.stem + p.suffix
            self.lr_files.append(self.lr_dir / lr_name)

    def __len__(self) -> int:
        return len(self.hr_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hr_path = self.hr_files[idx]
        lr_path = self.lr_files[idx]
        if not lr_path.exists():
            raise FileNotFoundError(f"LR file not found for {hr_path.name}: expected {lr_path}")

        hr_img = Image.open(hr_path).convert("L")
        lr_img = Image.open(lr_path).convert("L")

        hr = _to_chw_float01_gray(hr_img)  # [1,H,W]
        lr = _to_chw_float01_gray(lr_img)  # [1,h,w]

        crop = int(self.cfg.hr_crop)
        ctx = int(self.cfg.ctx_crop)
        crop_out = crop // 2

        hr, (hr_pad_top, hr_pad_left), (H0, W0) = _pad_tensor_to_min_hw_black_with_meta(hr, crop_out, crop_out)

        H4, W4 = int(hr.shape[1]), int(hr.shape[2])
        valid_full = torch.zeros((1, H4, W4), dtype=torch.float32)
        valid_full[:, hr_pad_top:hr_pad_top + H0, hr_pad_left:hr_pad_left + W0] = 1.0

        lr_up = F.interpolate(
            lr.unsqueeze(0),
            size=(H4 * 2, W4 * 2),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        lr_up_x4 = F.interpolate(
            lr.unsqueeze(0),
            size=(H4, W4),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        if self.split == "train":
            y0 = self.rng.randint(0, H4 - crop_out)
            x0 = self.rng.randint(0, W4 - crop_out)
        else:
            y0 = (H4 - crop_out) // 2
            x0 = (W4 - crop_out) // 2

        valid_crop = _crop_hw(valid_full, y0, x0, crop_out, crop_out)

        y0_8 = y0 * 2
        x0_8 = x0 * 2

        x_hr_1 = _extract_with_black_pad(lr_up, y0_8, x0_8, crop, crop)
        y_1 = _crop_hw(hr, y0, x0, crop_out, crop_out)
        x_base_1 = _crop_hw(lr_up_x4, y0, x0, crop_out, crop_out)

        cy = y0 + crop_out // 2
        cx = x0 + crop_out // 2
        cy8 = cy * 2
        cx8 = cx * 2
        ctx_y0 = cy8 - ctx // 2
        ctx_x0 = cx8 - ctx // 2
        ctx_patch_1 = _extract_with_black_pad(lr_up, ctx_y0, ctx_x0, ctx, ctx)

        x_lr_1 = F.interpolate(
            ctx_patch_1.unsqueeze(0),
            size=(crop, crop),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        if self.split == "train":
            x_hr_1, x_lr_1, y_1, x_base_1, valid_crop = _spatial_augment(
                x_hr_1, x_lr_1, y_1, x_base_1, valid_crop, self.rng
            )

        valid_bbox = _bbox_from_valid_mask(valid_crop)

        mean = float(self.cfg.normalize_mean)
        std = float(self.cfg.normalize_std)

        x_hr_1_n = _normalize(x_hr_1, mean, std)
        x_lr_1_n = _normalize(x_lr_1, mean, std)
        y_1_n = _normalize(y_1, mean, std)

        return {
            "x_hr": x_hr_1_n,
            "x_lr": x_lr_1_n,
            "y": y_1_n,
            "x_hr_base_01": x_base_1,
            "y_01": y_1,
            "valid_mask": valid_crop,
            "valid_bbox": valid_bbox,
            "meta_yx": torch.tensor([y0, x0], dtype=torch.int64),
            "meta_hw": torch.tensor([H4, W4], dtype=torch.int64),
            "name": hr_path.name,
        }
