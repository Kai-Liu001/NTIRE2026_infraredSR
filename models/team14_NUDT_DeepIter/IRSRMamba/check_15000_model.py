import torch

# 加载 15000 模型
model_path = '/hpcfs/fhome/competitor01/lhx/IRSRMamba/net_g_15000.pth'
checkpoint = torch.load(model_path, map_location='cpu')

print("Model keys:")
print(list(checkpoint['params'].keys())[:50])  # 打印前50个键
print("\nTotal number of keys:", len(checkpoint['params']))
