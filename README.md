# [NTIRE 2026 Challenge on Remote Sensing Infrared Image Super-Resolution (x4)](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)

## About the Challenge

The challenge is part of the First NTIRE Workshop at CVPR 2026, focusing on **remote sensing infrared image super-resolution**. Participants are required to recover high-resolution remote sensing infrared images from low-resolution inputs with a 4× upscaling factor.

**Single comprehensive evaluation track:**
- **Comprehensive Fidelity Track**: ranks methods by a combined score of pixel accuracy (PSNR) and structural similarity (SSIM), suitable for practical remote sensing infrared image applications.

## Notice

All submitted code must follow the format defined in this repository. Submissions that do not follow the required format may be rejected during the final evaluation stage.

After the challenge ends, we will release all submitted code as open-source for reproducibility. If you would like your model to remain confidential, please contact the organizers in advance.

## Challenge results

- **Valid submissions** are ranked; late submissions are shown below the line but excluded from the official leaderboard.
- **Evaluation set:** all scores are measured on the **InfaredSR-test (222 remote sensing infrared images)**.
- **Ranking Metric**:
  $$\text{Score} = \text{PSNR} + 20 \times \text{SSIM}$$
- Scores are computed on the **intensity channel of infrared images** with 4-px border shave.
- Higher Score indicates better performance.

![BQACAgUAAyEGAASHRsPbAAETKE9p33no6bJwxsp5cUfn5dhh0a-HlwACTSQAAhLfAAFXmqhR6JKcZws7BA.png](https://img.remit.ee/api/file/BQACAgUAAyEGAASHRsPbAAETKE9p33no6bJwxsp5cUfn5dhh0a-HlwACTSQAAhLfAAFXmqhR6JKcZws7BA.png)

## About this repository

This repository summarizes the solutions submitted by the participants during the challenge. The model script and the pre-trained weight parameters are provided in the [models](./models) and [model_zoo](./model_zoo) folders. Each team is assigned a number according to the submission time of the solution. You can find the correspondence between the number and team in [test.select_model](./test.py). Some participants would like to keep their models confidential. Thus, those models are not included in this repository.

## How to test the model?

1. `git clone https://github.com/Kai-Liu001/NTIRE2026_infraredSR.git`
   
2. Select the model you would like to test:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test.py --valid_dir [path to val data dir] --test_dir [path to test data dir] --save_dir [path to your save dir] --model_id 0
    ```
    - You can use either `--valid_dir`, or `--test_dir`, or both of them. Be sure to change the directories `--valid_dir`/`--test_dir` and `--save_dir`.

## How to add your model to this baseline?

> [!IMPORTANT]
>
> **🚨 Submissions that do not follow the official format will be rejected.**

1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1TR5yivh3o2FeALDMKvRtxOQpkDrk61LNNZVlM1vG5u4/edit?usp=sharing) and get your team ID.
2. Put your the code of your model in folder:  `./models/[Your_Team_ID]_[Your_Model_Name]`

   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02
3. Put the pretrained model in folder: `./model_zoo/[Your_Team_ID]_[Your_Model_Name]`

   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02
   - Note: Please provide a download link for the pretrained model, if the file size exceeds **100 MB**. Put the link in `./model_zoo/[Your_Team_ID]_[Your_Model_Name]/[Your_Team_ID]_[Your_Model_Name].txt`: e.g. [team00_dat.txt](./model_zoo/team00_dat/team00_dat.txt)
4. Add your model to the model loader `test.py` as follows:

   - Edit the `else` to `elif` in [test.py](./test.py#L24), and then you can add your own model with model id.

   - `model_func` **must** be a function, which accept **4 params**. 

     - `model_dir`: the pretrained model. Participants are expected to save their pretrained model in `./model_zoo/` with in a folder named `[Your_Team_ID]_[Your_Model_Name]` (e.g., team00_dat). 

     - `input_path`: a folder contains several images in PNG format. 

     - `output_path`: a folder contains restored images in PNG format. Please follow the section Folder Structure. 

     - `device`: computation device.
5. Send us the command to download your code, e.g,

   - `git clone [Your repository link]`
   - We will add your code and model checkpoint to the repository after the challenge.

> [!TIP]
>
> Your model code does not need to be fully refactored to fit this repository. 
> Instead, you may add a lightweight external interface (e.g., `models/team00_DAT/io.py`) that wraps your existing code, while keeping the original implementation unchanged.
>
> Refer to previous NTIRE challenge implementations for examples: 
> https://github.com/zhengchen1999/NTIRE2025_ImageSR_x4/tree/main/models


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
