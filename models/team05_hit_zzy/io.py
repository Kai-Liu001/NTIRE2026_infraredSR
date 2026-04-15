import argparse
import inspect
import logging
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


_NTIRE_ROOT = Path(__file__).resolve().parents[2]
ntire_root = str(_NTIRE_ROOT)
if ntire_root not in sys.path:
    sys.path.insert(0, ntire_root)

from utils import utils_logger
from utils import utils_image as util

try:
    from .model import TEAM05_GUIDE_MAP_OPT, TEAM05_MODEL_CONFIG, TEAM05_SCALE, build_model
except ImportError:
    from model import TEAM05_GUIDE_MAP_OPT, TEAM05_MODEL_CONFIG, TEAM05_SCALE, build_model


TEAM05_SELF_ENSEMBLE = True


def _normalize_map(img, eps=1e-6):
    img = img.astype(np.float32, copy=False)
    min_val = float(img.min())
    max_val = float(img.max())
    if max_val - min_val < eps:
        return np.zeros_like(img, dtype=np.float32)
    return (img - min_val) / (max_val - min_val)


def _ensure_hwc(img):
    if img.ndim == 2:
        img = img[..., None]
    return img


def _to_gray(img, image_channels=3):
    img = _ensure_hwc(img)
    use_channels = min(int(image_channels), img.shape[2])
    img = img[:, :, :use_channels].astype(np.float32, copy=False)

    if img.shape[2] == 1:
        return img[:, :, 0]
    if img.shape[2] >= 3:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    return img.mean(axis=2)


def _generate_edge_map(img, edge_opt=None, image_channels=3):
    edge_opt = edge_opt or {}
    mode = str(edge_opt.get("type", edge_opt.get("mode", "sobel"))).lower()
    blur_sigma = float(edge_opt.get("blur_sigma", 0.0))
    gray = _to_gray(img, image_channels=image_channels)

    if mode == "canny":
        low = int(edge_opt.get("low_threshold", 50))
        high = int(edge_opt.get("high_threshold", 150))
        edge = cv2.Canny(np.clip(gray * 255.0, 0, 255).astype(np.uint8), low, high).astype(np.float32) / 255.0
    else:
        ksize = int(edge_opt.get("ksize", 3))
        ksize = max(1, ksize)
        if ksize % 2 == 0:
            ksize += 1
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
        edge = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        if blur_sigma > 0:
            edge = cv2.GaussianBlur(edge, (0, 0), blur_sigma)
        edge = _normalize_map(edge)

    if blur_sigma > 0 and mode == "canny":
        edge = cv2.GaussianBlur(edge, (0, 0), blur_sigma)
        edge = _normalize_map(edge)

    return _ensure_hwc(edge.astype(np.float32))


def _generate_heat_map(img, heat_opt=None, image_channels=3):
    heat_opt = heat_opt or {}
    mode = str(heat_opt.get("type", heat_opt.get("mode", "intensity"))).lower()
    blur_sigma = float(heat_opt.get("blur_sigma", 1.2))
    spread_sigma = float(heat_opt.get("spread_sigma", 0.0))
    use_clahe = bool(heat_opt.get("clahe", False))
    gray = _to_gray(img, image_channels=image_channels)

    if use_clahe:
        clip_limit = float(heat_opt.get("clahe_clip_limit", 2.0))
        tile_size = int(heat_opt.get("clahe_tile_grid_size", 8))
        tile_size = max(1, tile_size)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        gray = clahe.apply(np.clip(gray * 255.0, 0, 255).astype(np.uint8)).astype(np.float32) / 255.0

    if mode == "gradient":
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        heat = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    elif mode == "laplacian":
        heat = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    else:
        heat = gray

    if blur_sigma > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), blur_sigma)
    if spread_sigma > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), spread_sigma)

    heat = _normalize_map(heat)
    return _ensure_hwc(heat.astype(np.float32))


def _append_guidance_maps(img, guide_map_opt=None):
    guide_map_opt = guide_map_opt or {}
    if not bool(guide_map_opt.get("enabled", False)):
        return _ensure_hwc(img).astype(np.float32, copy=False)

    image_channels = int(guide_map_opt.get("image_channels", 3))
    enabled_maps = [str(name).lower() for name in guide_map_opt.get("maps", ["edge", "heat"])]

    img = _ensure_hwc(img).astype(np.float32, copy=False)
    guide_maps = []

    if "edge" in enabled_maps and guide_map_opt.get("edge", {}).get("enabled", True):
        guide_maps.append(_generate_edge_map(img, guide_map_opt.get("edge", {}), image_channels=image_channels))
    if "heat" in enabled_maps and guide_map_opt.get("heat", {}).get("enabled", True):
        guide_maps.append(_generate_heat_map(img, guide_map_opt.get("heat", {}), image_channels=image_channels))

    if not guide_maps:
        return img
    return np.concatenate([img] + guide_maps, axis=2).astype(np.float32, copy=False)


def _hwc_to_tensor_preserve_extra(img, image_channels=3):
    img = _ensure_hwc(img)
    if img.shape[2] >= 3 and int(image_channels) >= 3:
        head = img[:, :, :3][:, :, ::-1]
        tail = img[:, :, 3:]
        img = np.concatenate([head, tail], axis=2)
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()


def _load_state_dict(model_path):
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")

    if any(key.startswith("module.") for key in checkpoint):
        checkpoint = {
            key[7:] if key.startswith("module.") else key: value for key, value in checkpoint.items()
        }
    return checkpoint


def _needs_scale_arg(model):
    signature = inspect.signature(model.forward)
    params = list(signature.parameters.values())
    required_positional = [
        param
        for param in params
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and param.default is inspect.Parameter.empty
    ]
    return len(required_positional) >= 2


def _build_input_tensor(img, device):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]

    img = img.astype(np.float32) / 255.0
    img = _append_guidance_maps(img, TEAM05_GUIDE_MAP_OPT)
    img_tensor = _hwc_to_tensor_preserve_extra(
        img,
        image_channels=int(TEAM05_GUIDE_MAP_OPT.get("image_channels", 3)),
    ).unsqueeze(0)

    expected_channels = int(TEAM05_MODEL_CONFIG["in_chans"])
    if img_tensor.size(1) != expected_channels:
        raise ValueError(
            f"Input channel mismatch: expected {expected_channels}, got {img_tensor.size(1)}. "
            "Check TEAM05_GUIDE_MAP_OPT and model config."
        )
    return img_tensor.to(device)


def _forward_model(model, img_tensor, needs_scale_arg):
    if needs_scale_arg:
        return model(img_tensor, TEAM05_SCALE)
    return model(img_tensor)


def _augment(x, op):
    if op == "r0":
        return x
    if op == "r90":
        return x.transpose(2, 3).flip(2)
    if op == "r180":
        return x.flip(2).flip(3)
    if op == "r270":
        return x.transpose(2, 3).flip(3)
    if op == "r0_h":
        return x.flip(3)
    if op == "r90_h":
        return x.transpose(2, 3).flip(2).flip(3)
    if op == "r180_h":
        return x.flip(2)
    if op == "r270_h":
        return x.transpose(2, 3)
    raise ValueError(f"Unsupported augmentation op: {op}")


def _deaugment(x, op):
    if op == "r0":
        return x
    if op == "r90":
        return x.flip(2).transpose(2, 3)
    if op == "r180":
        return x.flip(3).flip(2)
    if op == "r270":
        return x.flip(3).transpose(2, 3)
    if op == "r0_h":
        return x.flip(3)
    if op == "r90_h":
        return x.flip(3).flip(2).transpose(2, 3)
    if op == "r180_h":
        return x.flip(2)
    if op == "r270_h":
        return x.transpose(2, 3)
    raise ValueError(f"Unsupported deaugmentation op: {op}")


def _forward_x8(model, img_tensor, needs_scale_arg):
    ops = ["r0", "r90", "r180", "r270", "r0_h", "r90_h", "r180_h", "r270_h"]
    outputs = []
    for op in ops:
        augmented = _augment(img_tensor, op)
        output = _forward_model(model, augmented, needs_scale_arg)
        outputs.append(_deaugment(output, op))
    return torch.stack(outputs, dim=0).mean(dim=0)


def run(model, input_path, output_path, device):
    if input_path.endswith("/"):
        input_path = input_path[:-1]

    util.mkdir(output_path)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    input_img_list = [
        os.path.join(input_path, file_name)
        for file_name in sorted(os.listdir(input_path))
        if os.path.splitext(file_name)[1].lower() in valid_exts
    ]
    if not input_img_list:
        raise RuntimeError(f"No input images found in {input_path}")

    needs_scale_arg = _needs_scale_arg(model)

    with torch.no_grad():
        for img_path in input_img_list:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Failed to read image: {img_path}")

            img_tensor = _build_input_tensor(img, device)
            if TEAM05_SELF_ENSEMBLE:
                img_sr = _forward_x8(model, img_tensor, needs_scale_arg)
            else:
                img_sr = _forward_model(model, img_tensor, needs_scale_arg)
            img_sr = util.tensor2uint(img_sr, data_range=1.0)
            util.imsave(img_sr, os.path.join(output_path, f"{img_name}.png"))


def main(model_dir, input_path, output_path, device=None):
    utils_logger.logger_info("NTIRE2026-IRSRMambaPlus", log_path="NTIRE2026-IRSRMambaPlus.log")
    logger = logging.getLogger("NTIRE2026-IRSRMambaPlus")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    model = build_model()
    state_dict = _load_state_dict(model_dir)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = model.to(device)

    logger.info("Loaded team05 IRSRMambaPlus checkpoint from %s", model_dir)
    logger.info("Self-ensemble x8 enabled: %s", TEAM05_SELF_ENSEMBLE)
    run(model, input_path, output_path, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("team05_hit_IRSRMambaPlus")
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--device", default=None, type=str)
    args = parser.parse_args()

    main(
        model_dir=args.model_dir,
        input_path=args.input_path,
        output_path=args.output_path,
        device=args.device,
    )
