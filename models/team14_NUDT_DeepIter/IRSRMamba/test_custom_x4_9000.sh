#!/bin/bash -l
#SBATCH -p qgpu_4090
#SBATCH --job-name=custom_test_x4_9000
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=24:00:00
#SBATCH --mem=64GB

module load cuda/11.8
module load gcc/9.5.0
module load MPI/mpich/4.1.2-gcc-9.5.0

export PATH="/hpcfs/fpublic/app/miniforge3/conda/condabin:$PATH"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate gpsmamba

echo "="*80
echo "开始测试自定义数据集 (9000 迭代模型)"
echo "="*80
echo ""

export PYTHONPATH=/hpcfs/fhome/competitor01/lhx/IRSRMamba:$PYTHONPATH
cd /hpcfs/fhome/competitor01/lhx/IRSRMamba

python basicsr/test.py -opt options/test/test_custom_x4_9000.yml

echo ""
echo "="*80
echo "测试完成!"
echo "="*80
