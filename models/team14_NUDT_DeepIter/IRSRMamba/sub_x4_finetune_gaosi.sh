#!/bin/bash -l
#SBATCH -p qgpu_4090            # 指定分区 (这里使用 4090 GPU)
#SBATCH --job-name=ft_IRSR_gaosi # 作业名称
#SBATCH --output=log/%j.out         # 标准输出文件 (%j 会被替换为作业 ID)
#SBATCH --error=log/%j.err          # 错误输出文件
#SBATCH --gres=gpu:1            # 申请 1 块 GPU
#SBATCH --cpus-per-gpu=4        # 每个 GPU 分配的 CPU 核心数
#SBATCH --time=96:00:00         # 最大运行时间
#SBATCH --mem=64GB              # 申请内存大小

# 1. 启动 GPU 监控 (可选)
# 每 10 秒记录一次 GPU 利用率，后台运行
nvidia-smi dmon -d 10 -s puct -o T > log/${SLURM_JOB_ID}.log &

# 2. 激活 Conda 环境
export PATH="/hpcfs/fpublic/app/miniforge3/conda/condabin:$PATH"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate gpsmamba

# 3. 执行 IRSRMamba 高斯下采样微调训练任务
echo "开始 IRSRMamba 高斯下采样微调训练 (4 倍超分辨率)"
echo "使用预训练权重：/hpcfs/fhome/competitor01/lhx/IRSRMamba/experiments/train_IRSR_x4/models/net_g_50000.pth"
echo "使用训练数据：/hpcfs/fhome/competitor01/lhx/IRSRMamba/score/train_gaosi (高斯下采样)"
echo "学习率：3e-6，总迭代次数：100,000"
python basicsr/train.py -opt options/train/finetune_IRSR_x4_gaosi.yml

echo "训练完成！"
