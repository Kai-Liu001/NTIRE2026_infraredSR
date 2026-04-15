#!/usr/bin/env bash
set -euo pipefail

# 本地一键流程：
# 1) 跑 GPSMamba (net_g_60000)
# 2) 跑 IRSRMamba (net_g_45000)
# 3) 用 ensemble_package_minimal 做 0.4/0.6 融合

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSISR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPS_ROOT="${RSISR_ROOT}/GPSMamba_2/GPSMamba"
IRSR_ROOT="${RSISR_ROOT}/IRSRMamba"
DATA_DIR="${RSISR_ROOT}/../../dataset_test/LR/X4"

GPS_MODEL="${RSISR_ROOT}/model_zoo/team14_ensemble/GPSMamba/net_g_60000.pth"
IRSR_MODEL="${RSISR_ROOT}/model_zoo/team14_ensemble/IRSRMamba/net_g_45000.pth"

GPS_OPT_TMP="${GPS_ROOT}/options/test/GPSMamba/test_GPSMamba_x4_bicubic_perturb_60000_local.yml"
IRSR_OPT_TMP="${IRSR_ROOT}/options/test/test_IRSR_ft_bic_per_15000_45000_custom_local.yml"

GPS_RESULT_DIR="${GPS_ROOT}/results/GPS_fine_tune_bicubic_perturb_60000_local/visualization/LR_Only_Inference"
IRSR_RESULT_DIR="${IRSR_ROOT}/results/IRSR_finetune_bicubic_perturb_15000_45000_local/visualization/LR_Only_Inference"
FUSION_OUTPUT_DIR="${RSISR_ROOT}/results/team14_ensemble_fused"

for p in "${DATA_DIR}" "${GPS_MODEL}" "${IRSR_MODEL}"; do
  if [[ ! -e "${p}" ]]; then
    echo "缺少必要路径: ${p}"
    exit 1
  fi
done

mkdir -p "${RSISR_ROOT}/results"

echo "[1/3] 生成 GPSMamba 推理结果..."
cat > "${GPS_OPT_TMP}" <<EOF
name: GPS_fine_tune_bicubic_perturb_60000_local
model_type: MambaIRModel_basicsr
scale: 4
num_gpu: auto
manual_seed: 10
datasets:
  test_lr_only:
    name: LR_Only_Inference
    type: SingleImageDataset
    dataroot_lq: ${DATA_DIR}
    io_backend:
      type: disk
network_g:
  type: GPSMamba
  upscale: 4
  in_chans: 3
  img_size: 64
  img_range: 1.
  embed_dim: 174
  d_state: 16
  depths: [8, 8, 8, 8, 8, 8]
  num_heads: [6, 6, 6, 6, 6, 6]
  window_size: 16
  inner_rank: 64
  num_tokens: 128
  convffn_kernel_size: 5
  mlp_ratio: 2.
  upsampler: 'pixelshuffle'
  resi_connection: '1conv'
path:
  pretrain_network_g: ${GPS_MODEL}
  strict_load_g: true
val:
  save_img: true
  suffix: ''
  metrics: null
EOF

export PYTHONPATH="${GPS_ROOT}:${PYTHONPATH:-}"
cd "${GPS_ROOT}"
python BasicSR/test.py -opt "${GPS_OPT_TMP}"

echo "[2/3] 生成 IRSRMamba 推理结果..."
cat > "${IRSR_OPT_TMP}" <<EOF
name: IRSR_finetune_bicubic_perturb_15000_45000_local
model_type: MambaIRModel
scale: 4
num_gpu: auto
manual_seed: 10
datasets:
  test_lr_only:
    name: LR_Only_Inference
    type: SingleImageDataset
    dataroot_lq: ${DATA_DIR}
    io_backend:
      type: disk
network_g:
  type: IRSRMamba
  upscale: 4
  in_chans: 3
  img_size: 64
  img_range: 1.
  d_state: 16
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  mlp_ratio: 2
  upsampler: 'pixelshuffle'
  resi_connection: '1conv'
path:
  pretrain_network_g: ${IRSR_MODEL}
  strict_load_g: false
val:
  save_img: true
  suffix: ''
  metrics: null
EOF

export PYTHONPATH="${IRSR_ROOT}:${PYTHONPATH:-}"
cd "${IRSR_ROOT}"
python basicsr/test.py -opt "${IRSR_OPT_TMP}"

echo "[3/3] 执行加权融合 (GPS=0.4, IRSR=0.6)..."
cd "${RSISR_ROOT}/ensemble_package_minimal"
python ensemble_fusion.py \
  --gps-dir "${GPS_RESULT_DIR}" \
  --irsr-dir "${IRSR_RESULT_DIR}" \
  --output-dir "${FUSION_OUTPUT_DIR}" \
  --gps-weight 0.4 \
  --irsr-weight 0.6

echo "完成: ${FUSION_OUTPUT_DIR}"

