#!/bin/bash -l
#SBATCH -p qgpu_a800
#SBATCH --job-name=inference_irsr_ensemble
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=24:00:00
#SBATCH --mem=64GB

module load cuda/11.8

export PATH="/hpcfs/fpublic/app/miniforge3/conda/condabin:$PATH"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate gpsmamba

echo "="*80
echo "IRSR Bicubic Perturb 15000 多模型融合推理"
echo "="*80
echo ""
echo "参与融合的模型:"
echo "  - 20000 iter: /hpcfs/fhome/competitor01/lhx/IRSRMamba/experiments/finetune_IRSR_x4_bicubic_perturb_15000/models/net_g_20000.pth"
echo "  - 25000 iter: /hpcfs/fhome/competitor01/lhx/IRSRMamba/experiments/finetune_IRSR_x4_bicubic_perturb_15000/models/net_g_25000.pth (Best PSNR)"
echo "  - 30000 iter: /hpcfs/fhome/competitor01/lhx/IRSRMamba/experiments/finetune_IRSR_x4_bicubic_perturb_15000/models/net_g_30000.pth"
echo ""
echo "输入数据: /hpcfs/fhome/competitor01/lhx/testLR_X4/X4"
echo "融合策略: 基于PSNR的加权平均"
echo "="*80
echo ""

export PYTHONPATH=/hpcfs/fhome/competitor01/lhx/IRSRMamba_5:$PYTHONPATH


python inference_irsr_bicubic_perturb_ensemble.py

echo ""
echo "="*80
echo "融合完成!"
echo "="*80
