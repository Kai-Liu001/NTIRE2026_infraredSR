"""
Ensemble of 2 HAT-L models with TTA (8 geometric transforms) for
NTIRE 2026 Remote Sensing Infrared Image Super-Resolution (x4).

Best Score: 54.2287 (PSNR=35.81, SSIM=0.9207)
Weighted ensemble: 45% S1 + 55% BSR100K

Models:
  - model_s1.pth: HAT-L fine-tuned with L1+SSIM loss, lr=2e-5, 30K iters, seed=42
  - model_bsr100k.pth: HAT-L trained with basicsr, L1 loss, lr=2e-4, 100K iters, seed=42

Both models use HAT-L (40.8M params) pretrained on ImageNet+DF2K for x4 SR,
then fine-tuned on the competition's infrared remote sensing training data.
"""
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.team10_HAT_Ensemble.hat_arch import HAT


def create_hat_l():
    """Create HAT-L model for x4 SR."""
    return HAT(
        img_size=64, patch_size=1, in_chans=3, embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        window_size=16, compress_ratio=3, squeeze_factor=30,
        conv_scale=0.01, overlap_ratio=0.5, mlp_ratio=2,
        upscale=4, img_range=1.0, upsampler='pixelshuffle',
        resi_connection='1conv',
    )


def load_model(checkpoint_path, device):
    """Load a HAT-L model from checkpoint."""
    model = create_hat_l()
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'params_ema' in ckpt:
        model.load_state_dict(ckpt['params_ema'], strict=True)
    elif 'params' in ckpt:
        model.load_state_dict(ckpt['params'], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def pad_to_multiple(img, multiple=16):
    """Pad image to be divisible by multiple."""
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return img, h, w


def tta_forward(model, img, scale=4):
    """TTA with 8 geometric transforms (4 rotations x 2 flips)."""
    results = []
    for rot in range(4):
        for flip in [False, True]:
            x = img.clone()
            if flip:
                x = torch.flip(x, dims=[3])
            if rot > 0:
                x = torch.rot90(x, k=rot, dims=[2, 3])

            x, orig_h, orig_w = pad_to_multiple(x, 16)

            with torch.no_grad():
                out = model(x)

            out = out[:, :, :orig_h * scale, :orig_w * scale]

            if rot > 0:
                out = torch.rot90(out, k=4 - rot, dims=[2, 3])
            if flip:
                out = torch.flip(out, dims=[3])

            results.append(out.float())

    return torch.stack(results).mean(dim=0)


def main(model_dir, input_path, output_path, device=None):
    """
    Main entry point following the official interface.

    Args:
        model_dir: Path to model_zoo/team10_HAT_Ensemble/ directory
        input_path: Path to LR input images directory
        output_path: Path to save SR output images
        device: CUDA device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running HAT-L Ensemble + TTA on device: {device}")

    # Load two models
    ckpt1 = os.path.join(model_dir, 'model_s1.pth')
    ckpt2 = os.path.join(model_dir, 'model_bsr100k.pth')

    print(f"Loading model 1: {ckpt1}")
    model1 = load_model(ckpt1, device)
    print(f"Loading model 2: {ckpt2}")
    model2 = load_model(ckpt2, device)

    # Process images
    input_imgs = sorted(glob.glob(os.path.join(input_path, '*.[pP][nN][gG]')))
    os.makedirs(output_path, exist_ok=True)
    print(f"Processing {len(input_imgs)} images...")

    for i, img_path in enumerate(input_imgs):
        img_name = os.path.basename(img_path)

        # Load image
        img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # TTA forward for each model
        sr1 = tta_forward(model1, img_t, scale=4)
        sr2 = tta_forward(model2, img_t, scale=4)

        # Weighted ensemble (w1=0.45 for S1, w2=0.55 for BSR100K)
        sr = 0.45 * sr1 + 0.55 * sr2

        # Save output
        sr = sr.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        sr = (sr * 255).round().astype(np.uint8)
        Image.fromarray(sr).save(os.path.join(output_path, img_name))

        if (i + 1) % 50 == 0 or (i + 1) == len(input_imgs):
            print(f"  [{i+1}/{len(input_imgs)}] {img_name}")

    print(f"Done! Results saved to: {output_path}")
