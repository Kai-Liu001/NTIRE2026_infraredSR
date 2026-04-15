import torch

# 加载训练好的模型
model_path = '/hpcfs/fhome/competitor01/lhx/IRSRMamba/experiments/finetune_IRSR_x4_bicubic_perturb/models/net_g_40000.pth'
checkpoint = torch.load(model_path, map_location='cpu')

print("Model keys:")
print(list(checkpoint['params'].keys())[:50])  # 打印前50个键
print("\nTotal number of keys:", len(checkpoint['params']))

# 也检查一下原始预训练模型
pretrain_path = '/hpcfs/fhome/competitor01/lhx/IRSRMamba/experiments/finetune_IRSR_x4/models/net_g_50000.pth'
pretrain_checkpoint = torch.load(pretrain_path, map_location='cpu')

print("\n\nPretrain model keys:")
print(list(pretrain_checkpoint['params'].keys())[:50])  # 打印前50个键
print("\nTotal number of keys:", len(pretrain_checkpoint['params']))
