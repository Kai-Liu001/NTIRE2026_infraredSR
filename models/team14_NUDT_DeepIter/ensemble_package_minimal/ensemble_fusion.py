import os
import numpy as np
from PIL import Image
import re
import gc


def get_image_id(filename):
    match = re.match(r'^(\d+)_', filename)
    return match.group(1) if match else None


def generate_weighted_ensemble(gps_input_dir, irsr_input_dir, output_dir, weight_gps=0.4, weight_irsr=0.6):
    os.makedirs(output_dir, exist_ok=True)
    
    gps_images = {}
    for filename in os.listdir(gps_input_dir):
        img_id = get_image_id(filename)
        if img_id:
            gps_images[img_id] = filename
    
    irsr_images = {}
    for filename in os.listdir(irsr_input_dir):
        img_id = get_image_id(filename)
        if img_id:
            irsr_images[img_id] = filename
    
    common_ids = sorted(set(gps_images.keys()) & set(irsr_images.keys()))
    
    for img_id in common_ids:
        gps_path = os.path.join(gps_input_dir, gps_images[img_id])
        irsr_path = os.path.join(irsr_input_dir, irsr_images[img_id])
        
        gps_img = Image.open(gps_path).convert('RGB')
        irsr_img = Image.open(irsr_path).convert('RGB')
        
        gps_np = np.array(gps_img, dtype=np.float32)
        irsr_np = np.array(irsr_img, dtype=np.float32)
        
        ensemble_np = gps_np * weight_gps + irsr_np * weight_irsr
        ensemble_np = np.clip(ensemble_np, 0, 255).astype(np.uint8)
        
        result_img = Image.fromarray(ensemble_np)
        output_filename = f"{img_id}_GPS_fine_tune_mixed_perturb_75000.png"
        result_img.save(os.path.join(output_dir, output_filename))
        
        del gps_img, irsr_img, gps_np, irsr_np, ensemble_np, result_img
        gc.collect()
    
    print(f"融合完成！生成了 {len(common_ids)} 张图像到 {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='加权模型融合脚本')
    parser.add_argument('--gps-dir', required=True, help='GPSMamba推理结果目录')
    parser.add_argument('--irsr-dir', required=True, help='IRSRMamba推理结果目录')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--gps-weight', type=float, default=0.4, help='GPSMamba权重 (默认: 0.4)')
    parser.add_argument('--irsr-weight', type=float, default=0.6, help='IRSRMamba权重 (默认: 0.6)')
    
    args = parser.parse_args()
    
    generate_weighted_ensemble(
        args.gps_dir,
        args.irsr_dir,
        args.output_dir,
        args.gps_weight,
        args.irsr_weight
    )
