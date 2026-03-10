# [NTIRE 2026 Challenge on Remote Sensing Infrared Image Super-Resolution (x4)](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)

[![ntire](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fraw.githubusercontent.com%2Fzhengchen1999%2FNTIRE2026_RemoteSensingIR_SR_x4%2Fmain%2Ffigs%2Fdiamond_badge.json)](https://www.cvlai.net/ntire/2026/)
[![page](https://img.shields.io/badge/Project-Page-blue?logo=github&logoSvg)](https://ntire-sr.github.io/)
[![CVPRW](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fraw.githubusercontent.com%2Fzhengchen1999%2FNTIRE2026_RemoteSensingIR_SR_x4%2Fmain%2Ffigs%2Fcvf_badge.json)](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/html/Chen_NTIRE_2026_Challenge_on_Remote_Sensing_Infrared_Image_Super-Resolution_x4_Methods_and_Results_CVPRW_2026_paper.html)
[![arXiv](https://img.shields.io/badge/Report-arXiv-red?logo=arxiv&logoSvg)](https://arxiv.org/pdf/2604.14582)
[![supp](https://img.shields.io/badge/Supplementary-Paper-orange.svg)](https://github.com/zhengchen1999/NTIRE2026_RemoteSensingIR_SR_x4/releases/download/Supp/NTIRE.2026.Remote.Sensing.Infrared.Image.Super.Resolution.x4.Supplementary.pdf)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=zhengchen1999.NTIRE2026_RemoteSensingIR_SR_x4&right_color=violet)](https://github.com/zhengchen1999/NTIRE2026_RemoteSensingIR_SR_x4)
[![GitHub Stars](https://img.shields.io/github/stars/zhengchen1999/NTIRE2026_RemoteSensingIR_SR_x4?style=social)](https://github.com/zhengchen1999/NTIRE2026_RemoteSensingIR_SR_x4)

## About the Challenge

The challenge is part of the 11th NTIRE Workshop at CVPR 2026, focusing on **remote sensing infrared image super-resolution**. Participants are required to recover high-resolution remote sensing infrared images from low-resolution inputs with a 4× upscaling factor.

**Single comprehensive evaluation track:**
- **Comprehensive Fidelity Track**: ranks methods by a combined score of pixel accuracy (PSNR) and structural similarity (SSIM), suitable for practical remote sensing infrared image applications.

## Challenge results

- **Valid submissions** are ranked; late submissions are shown below the line but excluded from the official leaderboard.
- **Evaluation set:** all scores are measured on the **InfaredSR-test (222 remote sensing infrared images)**.
- **Ranking Metric**:
  $$\text{Score} = \text{PSNR} + 20 \times \text{SSIM}$$
- Scores are computed on the **intensity channel of infrared images** with 4-px border shave.
- Higher Score indicates better performance.

## Certificates

The top three teams by the comprehensive Score have received **NTIRE 2026 Remote Sensing Infrared Image SR (×4) Challenge certificates**:  

1. [Team Name 1]  
2. [Team Name 2]  
3. [Team Name 3]  

All certificates can be downloaded from [Google Drive](https://drive.google.com/file/d/1XXXXXXXXX/view?usp=sharing).

## About this repository

This repository summarizes the solutions submitted by the participants during the challenge. The model script and the pre-trained weight parameters are provided in the [models](./models) and [model_zoo](./model_zoo) folders. Each team is assigned a number according to the submission time of the solution. You can find the correspondence between the number and team in [test.select_model](./test.py). Some participants would like to keep their models confidential. Thus, those models are not included in this repository.

## How to test the model?

1. `https://github.com/Kai-Liu001/NTIRE2026_infraredSR.git`
2. Download the model weights from:

    - [Baidu Pan](https://pan.baidu.com/s/1XXXXXXXXX?pwd=RSIR) (validation code: **InfaredSR**)
    - [Google Drive](https://drive.google.com/drive/folders/1XXXXXXXXX?usp=drive_link)

    Put the downloaded weights in the `./model_zoo` folder.
3. Select the model you would like to test:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test.py --valid_dir [path to val data dir] --test_dir [path to test data dir] --save_dir [path to your save dir] --model_id 0
    ```
    - You can use either `--valid_dir`, or `--test_dir`, or both of them. Be sure to change the directories `--valid_dir`/`--test_dir` and `--save_dir`.
    - We provide a baseline (team00): DAT-IR (infrared-adapted DAT, default). Switch models (default is DAT-IR) by commenting the code in [test.py](./test.py#L19).
4. We also provide the output of each team from:

    - [Baidu Pan](https://pan.baidu.com/s/1XXXXXXXXX?pwd=RSIR) (validation code: **InfaredSR**)
    - [Google Drive](https://drive.google.com/drive/folders/1XXXXXXXXX?usp=drive_link)

    You can directly download the output of each team and evaluate the model using the provided script.
5. Some methods cannot be integrated into our codebase. We provide their instructions in the corresponding folder. If you still fail to test the model, please contact the team leaders. Their contact information is as follows:

| Index |       Team      |            Leader            |              Email              |
|:-----:|:---------------:|:----------------------------:|:-------------------------------:|
|   1   | [Team 1]        | [Leader Name 1]              | [Email 1]                       |
|   2   | [Team 2]        | [Leader Name 2]              | [Email 2]                       |
|   3   | [Team 3]        | [Leader Name 3]              | [Email 3]                       |
|   4   | [Team 4]        | [Leader Name 4]              | [Email 4]                       |
|   5   | [Team 5]        | [Leader Name 5]              | [Email 5]                       |
|   6   | [Team 6]        | [Leader Name 6]              | [Email 6]                       |
|   7   | [Team 7]        | [Leader Name 7]              | [Email 7]                       |
|   8   | [Team 8]        | [Leader Name 8]              | [Email 8]                       |
|   9   | [Team 9]        | [Leader Name 9]              | [Email 9]                       |
|  10   | [Team 10]       | [Leader Name 10]             | [Email 10]                      |
|  11   | [Team 11]       | [Leader Name 11]             | [Email 11]                      |
|  12   | [Team 12]       | [Leader Name 12]             | [Email 12]                      |
|  13   | [Team 13]       | [Leader Name 13]             | [Email 13]                      |
|  14   | [Team 14]       | [Leader Name 14]             | [Email 14]                      |
|  15   | [Team 15]       | [Leader Name 15]             | [Email 15]                      |
|  16   | [Team 16]       | [Leader Name 16]             | [Email 16]                      |
|  17   | [Team 17]       | [Leader Name 17]             | [Email 17]                      |
|  18   | [Team 18]       | [Leader Name 18]             | [Email 18]                      |
|  19   | [Team 19]       | [Leader Name 19]             | [Email 19]                      |
|  20   | [Team 20]       | [Leader Name 20]             | [Email 20]                      |
|  21   | [Team 21]       | [Leader Name 21]             | [Email 21]                      |
|  22   | [Team 22]       | [Leader Name 22]             | [Email 22]                      |
|  23   | [Team 23]       | [Leader Name 23]             | [Email 23]                      |
|  24   | [Team 24]       | [Leader Name 24]             | [Email 24]                      |
|  25   | [Team 25]       | [Leader Name 25]             | [Email 25]                      |
|  26   | [Team 26]       | [Leader Name 26]             | [Email 26]                      |

## How to eval images using IQA metrics?

### Environments

```sh
conda create -n NTIRE-InfaredSR python=3.8
conda activate NTIRE-InfaredSR
pip install -r requirements.txt
```

### Folder Structure
```
test_dir
├── HR
│   ├── 0901.png
│   ├── 0902.png
│   ├── ...
├── LQ
│   ├── 0901x4.png
│   ├── 0902x4.png
│   ├── ...
    
output_dir
├── 0901x4.png
├── 0902x4.png
├──...
```

### Command to calculate metrics

```sh
python eval.py \
--output_folder "/path/to/your/output_dir" \
--target_folder "/path/to/test_dir/HR" \
--metrics_save_path "./IQA_results" \
--gpu_ids 0 \
```

The `eval.py` file accepts the following 5 parameters:
- `output_folder`: Path where the restored infrared images are saved.
- `target_folder`: Path to the HR infrared images in the `test` dataset. This is used to calculate FR-IQA metrics.
- `metrics_save_path`: Directory where the evaluation metrics will be saved.
- `gpu_ids`: Computation devices. For multi-GPU setups, use the format `0,1,2,3`.

### Final Ranking Score

The official ranking is determined by the comprehensive score:
$$\text{Score} = \text{PSNR} + 20 \times \text{SSIM}$$

All metrics are averaged over the test set. Higher Score = better rank.

## NTIRE Remote Sensing Infrared Image SR ×4 Challenge Series

Code repositories and accompanying technical report PDFs for each edition:  

- **NTIRE 2026**: [CODE](https://github.com/zhengchen1999/NTIRE2026_RemoteSensingIR_SR_x4) | [PDF](https://arxiv.org/pdf/2604.14582)  

## Citation

If you find the code helpful in your research or work, please cite the following paper(s).

```
@inproceedings{ntire2026rsirsrx4,
  title={NTIRE 2026 Challenge on Remote Sensing Infrared Image Super-Resolution (x4): Methods and Results},
  author={Chen, Zheng and Liu, Kai and Gong, Jue and Wang, Jingkai and Sun, Lei and Wu, Zongwei and Timofte, Radu and Zhang, Yulun and others},
  booktitle={CVPRW},
  year={2026}
}

@inproceedings{ntire2025srx4,
  title={NTIRE 2025 Challenge on Image Super-Resolution (x4): Methods and Results},
  author={Chen, Zheng and Liu, Kai and Gong, Jue and Wang, Jingkai and Sun, Lei and Wu, Zongwei and Timofte, Radu and Zhang, Yulun and others},
  booktitle={CVPRW},
  year={2025}
}

@inproceedings{ntire2024srx4,
  title={Ntire 2024 challenge on image super-resolution (x4): Methods and results},
  author={Chen, Zheng and Wu, Zongwei and Zamfir, Eduard and Zhang, Kai and Zhang, Yulun and Timofte, Radu and others},
  booktitle={CVPRW},
  year={2024}
}

@inproceedings{ntire2023srx4,
  title={NTIRE 2023 challenge on image super-resolution (x4): Methods and results},
  author={Zhang, Yulun and Zhang, Kai and Chen, Zheng and Li, Yawei and Timofte, Radu and others},
  booktitle={CVPRW},
  year={2023}
}
```

## License and Acknowledgement
This code repository is released under [MIT License](LICENSE).
