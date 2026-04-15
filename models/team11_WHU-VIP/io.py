import os
import torch
import glob
import cv2
import numpy as np
from utils import utils_image as util
from .model import HAT  


def forward(img_lq, model, tile=None, tile_overlap=32, scale=4):
    if tile is None:
        output = model(img_lq)
    else:
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        sf = scale
        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h * sf, w * sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                # 修正后的切片赋值逻辑
                E[
                    ...,
                    h_idx * sf : (h_idx + tile) * sf,
                    w_idx * sf : (w_idx + tile) * sf,
                ] = out_patch
                W[
                    ...,
                    h_idx * sf : (h_idx + tile) * sf,
                    w_idx * sf : (w_idx + tile) * sf,
                ] += out_patch_mask
        output = E.div_(W)
    return output


def ensemble_forward(img_lq, model, scale=4):
    print("Testing with self-ensemble...")

    def _transform(v, op):
        v2np = v.data.cpu().numpy()
        if op == "v":
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == "h":
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == "t":
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).to(img_lq.device)
        return ret

    # prepare augmented data
    lq_list = [img_lq]
    for tf in "v", "h", "t":
        lq_list.extend([_transform(t, tf) for t in lq_list])

    # inference
    model.eval()
    with torch.no_grad():
        out_list = [model(aug) for aug in lq_list]

    if isinstance(out_list[0], list):
        out_list = [out[-1] for out in out_list]

    # merge results
    for i in range(len(out_list)):
        if i > 3:
            out_list[i] = _transform(out_list[i], "t")
        if i % 4 > 1:
            out_list[i] = _transform(out_list[i], "h")
        if (i % 4) % 2 == 1:
            out_list[i] = _transform(out_list[i], "v")
    output = torch.cat(out_list, dim=0)

    output = output.mean(dim=0, keepdim=True)
    return output


def main(model_dir, input_path, output_path, device=None, ensemble=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HAT(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=180,
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    # -------------------------------------------------------

    # 加载权重
    print(f"Loading model from {model_dir}...")
    loadnet = torch.load(model_dir, map_location="cpu")

    # 自动识别不同的 Key 格式
    if "params_ema" in loadnet:
        model.load_state_dict(loadnet["params_ema"], strict=True)
    elif "params" in loadnet:
        model.load_state_dict(loadnet["params"], strict=True)
    else:
        model.load_state_dict(loadnet, strict=True)

    model.eval()
    model = model.to(device)

    # 推理逻辑
    tile = None  
    os.makedirs(output_path, exist_ok=True)
    img_list = sorted(glob.glob(os.path.join(input_path, "*")))

    print(f"Start processing {len(img_list)} images...")
    for img_path in img_list:
        img_name, ext = os.path.splitext(os.path.basename(img_path))
        img_lr = util.imread_uint(img_path, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, data_range=1)
        img_lr = img_lr.to(device)

        # --- Padding 处理 ---
        window_size = 16
        _, _, h_old, w_old = img_lr.size()
        h_pad = (
            (h_old // window_size + 1) * window_size - h_old
            if h_old % window_size != 0
            else 0
        )
        w_pad = (
            (w_old // window_size + 1) * window_size - w_old
            if w_old % window_size != 0
            else 0
        )
        img_lr = torch.nn.functional.pad(img_lr, (0, w_pad, 0, h_pad), mode="reflect")
        # ------------------------

        with torch.no_grad():
            if ensemble:
                img_sr = ensemble_forward(img_lr, model)
            else:
                img_sr = forward(img_lr, model, tile)

        scale = 4  # 你的放大倍数
        img_sr = img_sr[..., : h_old * scale, : w_old * scale]
        # ----------------------------------

        img_sr = util.tensor2uint(img_sr, data_range=1)
        util.imsave(img_sr, os.path.join(output_path, img_name + ext))
        print(f"Saved: {img_name + ext}")
