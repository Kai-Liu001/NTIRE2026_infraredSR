import os
import torch
from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp
import csv
import numpy as np

from utils import utils_image as util

def read_csv_to_dict(filename):
    data = {}

    with open(filename, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            key = row[csv_reader.fieldnames[0]]
            data[key] = {
                field: (float(value) if is_number(value) else value)
                for field, value in row.items() if field != csv_reader.fieldnames[0]
            }

    return data


def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


class IQA:
    """
    Image Quality Assessment class for NTIRE 2026 Remote Sensing Infrared Image SR Challenge.
    Calculates PSNR and SSIM metrics, with final score = PSNR + 20 * SSIM.
    """
    def __init__(self, device=None):
        self.device = device if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def calculate_values(self, output_image, target_image, output_path=None, target_path=None):
        """
        Calculate PSNR and SSIM for the given image pair.
        
        Args:
            output_image: Output image (SR result)
            target_image: Target image (HR ground truth)
            output_path: Path to output image (for util.cal_psnr_ssim)
            target_path: Path to target image (for util.cal_psnr_ssim)
        
        Returns:
            dict: Dictionary containing 'psnr', 'ssim', and 'score' (PSNR + 20 * SSIM)
        """
        result = {}
        
        # Calculate PSNR and SSIM using utils_image
        if output_path is not None and target_path is not None:
            psnr_value, ssim_value = util.cal_psnr_ssim(output_path, target_path)
            result['psnr'] = psnr_value
            result['ssim'] = ssim_value
            # # Final score: PSNR + 20 * SSIM
            # result['score'] = psnr_value + 20 * ssim_value
        
        return result


def calculate_iqa_for_partition(output_folder, target_folder, output_files, device, rank):
    """
    Calculate IQA metrics (PSNR, SSIM, Score) for a partition of images.
    Score = PSNR + 20 * SSIM (NTIRE 2026 Remote Sensing Infrared Image SR Challenge)
    """
    iqa = IQA(device=device)
    local_results = {}
    for output_file in tqdm(output_files, total=len(output_files), desc=f"Processing images on GPU {rank}"):
        if target_folder is not None:
            target_file = output_file.replace('x4', '')

        output_image_path = os.path.join(output_folder, output_file)
        output_image = Image.open(output_image_path)

        if target_folder is not None:
            target_image_path = os.path.join(target_folder, target_file)
            assert os.path.exists(target_image_path), f"No such path: {target_image_path}"

            target_image = Image.open(target_image_path)
        else:
            target_image = None

        # Calculate PSNR, SSIM and Score using the new IQA class
        values = iqa.calculate_values(output_image, target_image, output_image_path, target_image_path)
        if values is not None:
            local_results[output_file] = values

    return local_results


def main_worker(rank, gpu_id, output_folder, target_folder, output_files, return_dict, num_gpus):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    partition_size = len(output_files) // num_gpus
    start_idx = rank * partition_size
    end_idx = (rank + 1) * partition_size if rank != num_gpus - 1 else len(output_files)

    output_files_partition = output_files[start_idx:end_idx]

    local_results = calculate_iqa_for_partition(output_folder, target_folder, output_files_partition,
                                                device, rank)
    return_dict[rank] = local_results


import argparse

if __name__ == "__main__":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, default="output_dir")
    parser.add_argument("--target_folder", type=str, default="div2k-val/HR")
    parser.add_argument("--metrics_save_path", type=str, default="./IQA_results")
    parser.add_argument("--gpu_ids", type=str, default="0")
    args = parser.parse_args()

    resume_figs = []
    output_files = sorted([f for f in os.listdir(args.output_folder) if f.endswith('.png') and f not in resume_figs])
    if args.target_folder is not None:
        target_files = sorted(
            [f for f in os.listdir(args.target_folder) if f.endswith('.png') and f not in resume_figs])

        assert len(output_files) == len(target_files), \
            (f"The number of output images should be equal to the number of target images: "
             f"{len(output_files)} != {len(target_files)}")
    else:
        target_files = None

    manager = mp.Manager()
    return_dict = manager.dict()

    args.gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
    print(f"Using GPU: {args.gpu_ids}")
    num_gpus = len(args.gpu_ids)

    processes = []
    for rank, gpu_id in enumerate(args.gpu_ids):
        p = mp.Process(target=main_worker, args=(
        rank, gpu_id, args.output_folder, args.target_folder, output_files, return_dict, num_gpus))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = {}
    for rank in return_dict.keys():
        results.update(return_dict[rank])

    folder_name = os.path.basename(args.output_folder)
    parent_folder = os.path.dirname(args.output_folder)
    next_level_folder = os.path.basename(parent_folder)
    os.makedirs(args.metrics_save_path, exist_ok=True)
    average_results_filename = f"{args.metrics_save_path}/{next_level_folder}--{folder_name}.txt"
    results_filename = f"{args.metrics_save_path}/{next_level_folder}--{folder_name}.csv"

    if results:
        all_keys = set()
        for values in results.values():
            try:
                all_keys.update(values.keys())
            except Exception as e:
                print(f"Error: {e}")

        all_keys = sorted(all_keys)

        average_results = {}
        for key in all_keys:
            average_results[key] = np.mean([values.get(key, 0) for values in results.values()])

        # Calculate final score: Score = PSNR + 20 * SSIM (as per NTIRE 2026 Remote Sensing Infrared Image SR challenge)
        psnr_value = average_results.get('psnr', 0)
        ssim_value = average_results.get('ssim', 0)
        average_results['Total Score'] = psnr_value + 20 * ssim_value
        print(f"PSNR: {psnr_value}")
        print(f"SSIM: {ssim_value}")
        print(f"Final Score (PSNR + 20 * SSIM): {average_results['Total Score']}")

        print("Average:")
        print(average_results)
        
        with open(results_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename'] + list(all_keys))
            for filename, values in results.items():
                row = [filename] + [values.get(key, '') for key in all_keys]
                writer.writerow(row)
            print(f"IQA results have been saved to {results_filename} file")

        with open(average_results_filename, 'w') as f:
            for key, value in average_results.items():
                f.write(f"{key}: {value}\n")
            print(f"Average IQA results and Weighted Score have been saved to {average_results_filename} file")
