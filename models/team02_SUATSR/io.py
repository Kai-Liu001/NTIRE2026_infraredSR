import os
import logging
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils_logger
from utils import utils_image as util

from models.team02_WIRSR_TEFA_Net.model import WIRSR_TEFA_Net


# =====================================================================
# 灰度图 I/O 工具（与 convert_rgb_to_l 保持一致的 PIL 转换逻辑）
# =====================================================================
def load_grayscale(path: str):
    """
    读取图像并转为灰度图，返回 (float32 array [0,1], max_val)。
    转换逻辑与 convert_rgb_to_l 完全一致：
      - RGBA → RGB → L
      - RGB  → L
      - L    → 直接使用
    """
    img = Image.open(path)

    # ── 与 convert_rgb_to_l 一致的灰度转换逻辑 ──
    if img.mode != 'L':
        # 对于带透明通道的图片（RGBA），先转换为 RGB
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        # 转换为灰度图
        img = img.convert('L')

    arr = np.array(img)
    # 判断位深
    max_val = 65535.0 if arr.dtype == np.uint16 else 255.0
    return arr.astype(np.float32) / max_val, max_val


def save_grayscale(path: str, arr01: np.ndarray, max_val: float = 255.0):
    """
    保存灰度图，使用 PIL Image 保持与加载阶段库调用一致。
    """
    arr01 = np.clip(arr01, 0., 1.)
    if max_val > 255.0:
        out = (arr01 * max_val + 0.5).astype(np.uint16)
        Image.fromarray(out, mode='I;16').save(path)
    else:
        out = (arr01 * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(out, mode='L').save(path)


# =====================================================================
# 推理工具（Tile + TTA x8）
# =====================================================================
def tile_forward(model: nn.Module, lr: torch.Tensor, scale: int,
                 tile: int = 0, overlap: int = 16) -> torch.Tensor:
    """分块推理"""
    b, c, h, w = lr.shape
    if tile is None or tile <= 0 or (h <= tile and w <= tile):
        out = model(lr)
        if isinstance(out, tuple):
            out = out[0]
        return out
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("tile must be > overlap")
    out_t = torch.zeros((b, 1, h * scale, w * scale), device=lr.device, dtype=lr.dtype)
    wmap  = torch.zeros_like(out_t)
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y0 = min(y, h - tile)
            x0 = min(x, w - tile)
            patch = lr[:, :, y0:y0 + tile, x0:x0 + tile]
            p_out = model(patch)
            if isinstance(p_out, tuple):
                p_out = p_out[0]
            out_t[:, :, y0 * scale:(y0 + tile) * scale,
                        x0 * scale:(x0 + tile) * scale] += p_out
            wmap[:, :, y0 * scale:(y0 + tile) * scale,
                       x0 * scale:(x0 + tile) * scale] += 1.
    return out_t / wmap


def _aug_x8(x, k, flip):
    if flip:
        x = torch.flip(x, [-1])
    if k:
        x = torch.rot90(x, k, [-2, -1])
    return x


def _deaug_x8(x, k, flip):
    if k:
        x = torch.rot90(x, 4 - k, [-2, -1])
    if flip:
        x = torch.flip(x, [-1])
    return x


@torch.no_grad()
def forward_sr(model: nn.Module, lr: torch.Tensor, scale: int,
               tile: int = 0, overlap: int = 16, tta: bool = False) -> torch.Tensor:
    """带 TTA（8 种几何变换自集成）的推理"""
    if not tta:
        return tile_forward(model, lr, scale, tile, overlap)
    acc = None
    n = 0
    for flip in (False, True):
        for k in (0, 1, 2, 3):
            lr_a = _aug_x8(lr, k, flip)
            sr_a = tile_forward(model, lr_a, scale, tile, overlap)
            sr_a = _deaug_x8(sr_a, k, flip)
            acc  = sr_a if acc is None else acc + sr_a
            n   += 1
    return acc / float(n)


# =====================================================================
# 主推理入口
# =====================================================================
def run(model, data_path, save_path, tile, device, tta=True):
    """遍历输入文件夹，逐图推理并保存"""
    sf = 4

    if data_path.endswith('/'):
        data_path = data_path[:-1]

    # 支持的图片格式（与 convert_rgb_to_l 一致）
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif')

    input_img_list = []
    for fname in sorted(os.listdir(data_path)):
        if fname.lower().endswith(supported_formats):
            input_img_list.append(os.path.join(data_path, fname))

    os.makedirs(save_path, exist_ok=True)

    for i, img_path in enumerate(input_img_list):
        img_name = os.path.basename(img_path)
        print(f"[{i + 1}/{len(input_img_list)}] Processing: {img_name}")

        # 加载灰度图（PIL .convert('L') 逻辑）
        img_arr, max_val = load_grayscale(img_path)
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        img_tensor = img_tensor.to(device)

        # 推理（默认开启 TTA）
        with torch.no_grad():
            sr = forward_sr(model, img_tensor, scale=sf,
                            tile=tile, overlap=16, tta=tta)

        # 保存
        sr_np = sr.squeeze().cpu().numpy()
        save_grayscale(os.path.join(save_path, img_name), sr_np, max_val)


def main(model_dir, input_path, output_path, device=None):
    """
    模板要求的统一入口函数。
    model_dir:   权重文件路径
    input_path:  输入图像文件夹
    output_path: 输出保存文件夹
    device:      推理设备
    """
    utils_logger.logger_info("NTIRE2026-RemoteSensingIR-SRx4",
                             log_path="NTIRE2026-RemoteSensingIR-SRx4.log")
    logger = logging.getLogger("NTIRE2026-RemoteSensingIR-SRx4")

    # ── 基础设置 ──
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on device: {device}')

    # ── 构建模型（与训练时默认参数一致） ──
    model = WIRSR_TEFA_Net(
        in_ch=1,
        feat_ch=160,
        noise_feat_ch=64,
        n_hat=8,
        n_hfb=8,
        n_fiu=4,
        scale=4,
        ws=8,
        ov=2,
        nh=8,
    )

    # ── 加载权重（优先使用 EMA） ──
    ckpt = torch.load(model_dir, map_location=device)
    if isinstance(ckpt, dict):
        state = None
        if ckpt.get('ema') is not None:
            state = ckpt['ema']
        elif ckpt.get('model') is not None:
            state = ckpt['model']
        model.load_state_dict(state or ckpt, strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # ── 推理（TTA 已开启）──
    tile = None  # 不使用分块；如显存不足可设为如 128
    run(model, input_path, output_path, tile, device, tta=True)