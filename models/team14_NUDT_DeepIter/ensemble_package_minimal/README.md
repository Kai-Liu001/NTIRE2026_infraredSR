# ensemble_weighted_avg_top2 最小化部署包

## 概述

这是 ensemble_weighted_avg_top2 模型融合组件的最小化部署包，仅包含执行所需的核心文件。

## 目录结构

```
ensemble_package_minimal/
├── README.md                          # 本文档
├── requirements.txt                   # Python依赖
├── ensemble_fusion.py                # 核心融合脚本
```

## 快速使用


### 生成融合结果

如需重新生成，按以下步骤操作：

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行融合脚本：
```bash
python ensemble_fusion.py \
    --gps-dir /path/to/GPSMamba/results \
    --irsr-dir /path/to/IRSRMamba/results \
    --output-dir ./output
```

3. 可选参数：
```bash
--gps-weight 0.4    # GPSMamba权重（默认0.4）
--irsr-weight 0.6   # IRSRMamba权重（默认0.6）
```

## 依赖说明

仅需两个核心Python包：
- `numpy>=1.21.0` - 数值计算
- `Pillow>=8.0.0` - 图像处理

## 文件说明

### ensemble_fusion.py

核心融合脚本，功能包括：
- 从两个单模型推理结果读取图像
- 按指定权重进行加权平均融合
- 保存融合结果

### requirements.txt

最小化的Python依赖列表。

## 预生成结果

- 图像数量：222张
- 图像格式：PNG
- 权重配置：GPS=40%, IRSR=60%
- 命名格式：`{ID}_GPS_fine_tune_mixed_perturb_75000.png`

## 注意事项

1. 本包不包含单模型推理结果和训练模型文件
2. 如需重新生成，需自行准备GPSMamba和IRSRMamba的推理结果
3. 预生成结果可直接使用，无需运行融合脚本
