#!/bin/bash -l
#SBATCH -p qgpu_4090            # 指定分区 (这里使用 4090 GPU)
#SBATCH --job-name=ft_IRSR_gaosi_lr1e6 # 作业名称
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

# 3. 执行 IRSRMamba 优化学习率微调训练任务
echo "开始 IRSRMamba 优化学习率微调训练 (高斯下采样数据)"
echo "预训练模型：/hpcfs/fhome/competitor01/lhx/IRSRMamba/experiments/train_IRSR_x4/models/net_g_50000.pth"
echo "训练数据：/hpcfs/fhome/competitor01/lhx/IRSRMamba/score/train_gaosi (高斯下采样)"
echo "学习率：1e-6 (优化策略)"
echo "学习率衰减点：[15000, 30000, 50000, 70000]"
echo "总迭代次数：100,000"
echo "预期目标：达到 53.5+ 分"
python basicsr/train.py -opt options/train/finetune_IRSR_x4_gaosi_lr1e6.yml

echo "训练完成！"
