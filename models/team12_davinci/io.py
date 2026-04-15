import os.path
import logging
import torch
import json
import glob
import torch.nn.functional as F

from utils import utils_logger
from utils import utils_image as util

from models.team12_davinci.hat_arch import HAT


def forward(img_lq, models, tile=None, tile_overlap=32, scale=4, selfensemble=True):
    """
    Forward function with optional tiling, self-ensemble, and multi-model ensemble.
    
    Args:
        img_lq: Low-quality input image tensor
        models: List of HAT models (supports single model or model ensemble)
        tile: Tile size for processing large images (None for whole image)
        tile_overlap: Overlap size between tiles
        scale: Upscaling factor
        selfensemble: Whether to use self-ensemble (flipping augmentation)
    
    Returns:
        Super-resolved image tensor
    """
    # Support single model or list of models
    if not isinstance(models, list):
        models = [models]
    
    # Pad input image to be a multiple of window_size (16)
    window_size = 16
    mod_pad_h, mod_pad_w = 0, 0
    _, _, h_old, w_old = img_lq.size()
    if h_old % window_size != 0:
        mod_pad_h = window_size - h_old % window_size
    if w_old % window_size != 0:
        mod_pad_w = window_size - w_old % window_size
    img_lq = F.pad(img_lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    
    if tile is None:
        # test the image as a whole
        sr_img_list = []
        for model in models:
            if selfensemble:
                output = test_selfensemble(img_lq, model)
            else:
                output = model(img_lq)
            sr_img_list.append(output)
        
        # Multi-model ensemble: average outputs from all models
        if len(sr_img_list) > 1:
            sr_img = torch.cat(sr_img_list, dim=0)
            output = sr_img.mean(dim=0, keepdim=True)
        else:
            output = sr_img_list[0]
        
        # Remove padding from output
        _, _, h, w = output.size()
        output = output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
    else:
        # test the image tile by tile with self-ensemble and multi-model ensemble
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        tile_overlap = tile_overlap
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        
        # Aggregate outputs from all models
        E_list = []
        W_list = []
        
        for model in models:
            E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    if selfensemble:
                        out_patch = test_selfensemble(in_patch, model)
                    else:
                        out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
            
            output = E.div_(W)
            E_list.append(output)
        
        # Multi-model ensemble: average outputs from all models
        if len(E_list) > 1:
            E_all = torch.cat(E_list, dim=0)
            output = E_all.mean(dim=0, keepdim=True)
        else:
            output = E_list[0]

    return output


def test_selfensemble(lq, model):
    """Self-ensemble: use horizontal and vertical flips to improve results"""
    def _transform(v, op):
        v2np = v.data.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).to(v.device)
        return ret

    # prepare augmented data
    lq_list = [lq]
    for tf in 'v', 'h', 't':
        lq_list.extend([_transform(t, tf) for t in lq_list])

    # inference
    model.eval()
    with torch.no_grad():
        out_list = [model(aug) for aug in lq_list]

    # merge results
    for i in range(len(out_list)):
        if i > 3:
            out_list[i] = _transform(out_list[i], 't')
        if i % 4 > 1:
            out_list[i] = _transform(out_list[i], 'h')
        if i % 2 == 1:
            out_list[i] = _transform(out_list[i], 'v')
    output = torch.cat(out_list, dim=0)

    output = output.mean(dim=0, keepdim=True)
    return output


def run(models, data_path, save_path, tile, device, selfensemble=True):
    """
    Run inference on all images in data_path and save to save_path.
    
    Args:
        models: HAT model or list of HAT models (for ensemble)
        data_path: Path to input images
        save_path: Path to save output images
        tile: Tile size for processing large images (None for whole image)
        device: Torch device (cuda or cpu)
        selfensemble: Whether to use self-ensemble augmentation
    """
    data_range = 1.0
    sf = 4
    border = sf

    if data_path.endswith('/'):  # solve when path ends with /
        data_path = data_path[:-1]
    # scan all the jpg and png images
    input_img_list = sorted(glob.glob(os.path.join(data_path, '*.[jpJP][pnPN]*[gG]')))
    # save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    for i, img_lr in enumerate(input_img_list):

        # --------------------------------
        # (1) img_lr
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_lr))
        img_lr = util.imread_uint(img_lr, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, data_range)
        img_lr = img_lr.to(device)

        # --------------------------------
        # (2) img_sr
        # --------------------------------
        img_sr = forward(img_lr, models, tile, selfensemble=selfensemble)
        img_sr = util.tensor2uint(img_sr, data_range)

        util.imsave(img_sr, os.path.join(save_path, img_name+ext))


def main(model_dir, input_path, output_path, device=None, selfensemble=True):
    """
    Main function to load model and run inference.
    
    Args:
        model_dir: Path to model checkpoint file (or comma-separated list for ensemble)
        input_path: Path to input images directory
        output_path: Path to save output images
        device: Torch device (cuda or cpu), auto-detected if None
        selfensemble: Whether to use self-ensemble augmentation
    """
    utils_logger.logger_info("NTIRE2026-InfraredSR-HAT", log_path="NTIRE2026-InfraredSR-HAT.log")
    logger = logging.getLogger("NTIRE2026-InfraredSR-HAT")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Running on device: {device}')

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    
    # Support single model or ensemble models (comma-separated)
    model_paths = [path.strip() for path in model_dir.split(',')]
    print(f'Loading {len(model_paths)} model(s): {model_paths}')
    
    models = []
    for model_path in model_paths:
        print(f'Loading model from: {model_path}')
        model = HAT(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv')

        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)

        model.eval()
        tile = None
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
        models.append(model)
    
    # Run inference with model(s) - supports single model or multi-model ensemble
    run(models, input_path, output_path, tile, device, selfensemble)
