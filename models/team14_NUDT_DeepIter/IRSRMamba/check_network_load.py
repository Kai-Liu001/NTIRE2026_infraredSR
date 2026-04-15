import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from basicsr.archs import build_network
from basicsr.utils.options import parse_options

# 定义测试配置
opt = {
    'network_g': {
        'type': 'IRSRMamba',
        'upscale': 4,
        'in_chans': 3,
        'img_size': 64,
        'img_range': 1.,
        'd_state': 16,
        'depths': [8, 8, 8, 8, 8, 8],
        'embed_dim': 180,
        'mlp_ratio': 2,
        'upsampler': 'pixelshuffle',
        'resi_connection': '1conv'
    }
}

# 构建网络
print("Building network...")
net = build_network(opt['network_g'])

# 加载预训练模型
model_path = '/hpcfs/fhome/competitor01/lhx/IRSRMamba/experiments/finetune_IRSR_x4_bicubic_perturb/models/net_g_40000.pth'
checkpoint = torch.load(model_path, map_location='cpu')
state_dict = checkpoint['params']

print(f"\nNetwork has {len(net.state_dict().keys())} keys")
print(f"Checkpoint has {len(state_dict.keys())} keys")

# 检查差异
net_keys = set(net.state_dict().keys())
checkpoint_keys = set(state_dict.keys())

missing_in_net = checkpoint_keys - net_keys
missing_in_checkpoint = net_keys - checkpoint_keys

print(f"\nKeys in checkpoint but not in network: {len(missing_in_net)}")
if missing_in_net:
    print("First 20 keys:")
    for k in list(missing_in_net)[:20]:
        print(f"  {k}")

print(f"\nKeys in network but not in checkpoint: {len(missing_in_checkpoint)}")
if missing_in_checkpoint:
    print("First 20 keys:")
    for k in list(missing_in_checkpoint)[:20]:
        print(f"  {k}")

# 尝试加载
print("\nTrying to load state dict with strict=False...")
try:
    net.load_state_dict(state_dict, strict=False)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
