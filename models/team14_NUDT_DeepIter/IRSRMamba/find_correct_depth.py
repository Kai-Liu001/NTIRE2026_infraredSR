import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from basicsr.archs import build_network

# 尝试不同的 depths
depth_options = [
    [6, 6, 6, 6],
    [8, 8, 8, 8],
    [6, 6, 6, 6, 6, 6],
    [4, 4, 4, 4, 4, 4],
    [4, 4, 4, 4],
]

for depths in depth_options:
    print(f"\nTesting depths: {depths}")
    opt = {
        'network_g': {
            'type': 'IRSRMamba',
            'upscale': 4,
            'in_chans': 3,
            'img_size': 64,
            'img_range': 1.,
            'd_state': 16,
            'depths': depths,
            'embed_dim': 180,
            'mlp_ratio': 2,
            'upsampler': 'pixelshuffle',
            'resi_connection': '1conv'
        }
    }
    
    net = build_network(opt['network_g'])
    print(f"Number of keys: {len(net.state_dict().keys())}")
