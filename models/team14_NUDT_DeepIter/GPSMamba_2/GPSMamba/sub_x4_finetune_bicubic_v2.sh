#!/bin/bash -l
#SBATCH -p qgpu_a800
#SBATCH --job-name=ft_GPS_bicubic_v2
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=96:00:00
#SBATCH --mem=64GB

module load cuda/11.8
module load gcc/9.5.0
module load MPI/mpich/4.1.2-gcc-9.5.0

export PATH="/hpcfs/fpublic/app/miniforge3/conda/condabin:$PATH"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate gpsmamba

echo "开始训练：GPSMamba 双三次下采样"
echo "预训练模型：/hpcfs/fhome/competitor01/lhx/GPSMamba_2/net_g_x4_best.pth"
export PYTHONPATH=/hpcfs/fhome/competitor01/lhx/GPSMamba_2/GPSMamba:$PYTHONPATH
cd /hpcfs/fhome/competitor01/lhx/GPSMamba_2/GPSMamba
python basicsr/train.py -opt options/train/finetune_GPS_x4_bicubic_v2.yml

echo "训练完成！"
