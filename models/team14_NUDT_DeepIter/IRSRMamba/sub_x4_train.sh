#!/bin/bash -l
#SBATCH -p qgpu_a800            # 指定分区 (这里使用 A800 GPU)
#SBATCH --job-name=irsrmamba_x4 # 作业名称
#SBATCH --output=log/%j.out         # 标准输出文件 (%j 会被替换为作业 ID)
#SBATCH --error=log/%j.err          # 错误输出文件
#SBATCH --gres=gpu:1            # 申请 1 块 GPU
#SBATCH --cpus-per-gpu=4        # 每个 GPU 分配的 CPU 核心数
#SBATCH --time=96:00:00         # 最大运行时间
#SBATCH --mem=64GB              # 申请内存大小


# 1. 启动 GPU 监控 (可选)
# 每 10 秒记录一次 GPU 利用率，后台运行
nvidia-smi dmon -d 10 -s puct -o T > log/${SLURM_JOB_ID}.log &
module load cuda/11.8
module load gcc/9.5.0
module load MPI/mpich/4.1.2-gcc-9.5.0

# 2. 激活 Conda 环境
export PATH="/hpcfs/fpublic/app/miniforge3/conda/condabin:$PATH"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate gpsmamba

# 3. 执行训练任务
echo "开始训练 IRSRMamba 模型 (4倍超分辨率)"
python basicsr/train.py -opt options/train/train_IRSRMamba_score_x4.yml