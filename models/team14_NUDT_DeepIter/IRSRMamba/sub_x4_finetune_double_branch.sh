#!/bin/bash -l
#SBATCH -p qgpu_a800            # 指定分区 (这里使用 a800 GPU)
#SBATCH --job-name=ft_IRSR_double_branch # 作业名称
#SBATCH --output=log/%j.out         # 标准输出文件 (%j 会被替换为作业 ID)
#SBATCH --error=log/%j.err          # 错误输出文件
#SBATCH --gres=gpu:1            # 申请 1 块 GPU
#SBATCH --cpus-per-gpu=4        # 每个 GPU 分配的 CPU 核心数
#SBATCH --time=96:00:00         # 最大运行时间
#SBATCH --mem=64GB              # 申请内存大小

# 1. 加载 CUDA 和 MPI 模块
module load cuda/11.8
module load gcc/9.5.0
module load MPI/mpich/4.1.2-gcc-9.5.0

# 2. 激活 Conda 环境
export PATH="/hpcfs/fpublic/app/miniforge3/conda/condabin:$PATH"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate gpsmamba

# 3. 设置 CUDA 环境变量
export CUDA_HOME=/hpcfs/fpublic/app/cuda/11.8
export LD_LIBRARY_PATH=~/cuda_lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 5. 执行训练任务
echo "开始训练：IRSRMamba 双分支训练"
echo "预训练模型：/hpcfs/fhome/competitor01/lhx/IRSRMamba/experiments/train_IRSR_x4/models/net_g_50000.pth"
echo "训练配置：options/train/finetune_IRSR_x4_double_branch.yml"
export PYTHONPATH=/hpcfs/fhome/competitor01/lhx/IRSRMamba:$PYTHONPATH
cd /hpcfs/fhome/competitor01/lhx/IRSRMamba
python basicsr/train.py -opt options/train/finetune_IRSR_x4_double_branch.yml

echo "训练完成！"