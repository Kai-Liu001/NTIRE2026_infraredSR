#!/bin/bash -l
#SBATCH -p qgpu_4090
#SBATCH --job-name=test_IRSR_mixed_75000
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

echo "开始测试 IRSRMamba mixed_75000 模型 (4倍超分辨率)"
echo "模型路径：/hpcfs/fhome/competitor01/lhx/IRSRMamba/experiments/finetune_IRSR_x4_mixed/models/net_g_75000.pth"
echo "输入数据：/hpcfs/fhome/competitor01/lhx/testLR_X4/X4"
export PYTHONPATH=/hpcfs/fhome/competitor01/lhx/IRSRMamba:$PYTHONPATH
cd /hpcfs/fhome/competitor01/lhx/IRSRMamba
python basicsr/test.py -opt options/test/test_IRSR_mixed_75000_custom.yml

echo "测试完成！"

