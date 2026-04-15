#!/bin/bash -l
#SBATCH -p qgpu_4090            # 指定分区 (这里使用 4090 GPU)
#SBATCH --job-name=test_IRSR_gaosi # 作业名称
#SBATCH --output=log/%j.out         # 标准输出文件
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

# 2. 执行测试任务 - 使用高斯下采样最佳模型放大图像
echo "开始测试：使用 finetune_IRSR_x4_gaosi 的 net_g_50000.pth 模型"
echo "模型路径：/hpcfs/fhome/competitor01/lhx/IRSRMamba/experiments/finetune_IRSR_x4_gaosi/models/net_g_50000.pth"
echo "输入数据：/hpcfs/fhome/competitor01/lhx/IRSRMamba/score/train/val_LR_X4/X4"
echo "输出结果：/hpcfs/fhome/competitor01/lhx/IRSRMamba/results/IRSR_finetune_gaosi_50000/visualization"
export PYTHONPATH=/hpcfs/fhome/competitor01/lhx/IRSRMamba:$PYTHONPATH
python basicsr/test.py -opt options/test/test_IRSR_gaosi_50000.yml

echo "测试完成！"
