
# SADER模型
本仓库为**SADER**模型的官方实现，该方法通过通过引入全新的**MTCDN**网络和基于云层注意力损失函数和重采样，在遥感图像去云任务中展现出卓越性能。
## 实验结果 Experimental Results

<table>
<tr>
<th align="center">1. Sen2_MTC_New 定量对比</th>
<th align="center">2. SEN12MS-CR-TS(EA) 定量对比</th>
</tr>
<tr>
<td valign="top">

| 方法 (Method) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :---: | :---: | :---: |
| McGAN (2017) | 17.448 | 0.513 | 0.447 |
| Pix2Pix (2017) | 16.985 | 0.455 | 0.535 |
| AE (2018) | 15.100 | 0.441 | 0.602 |
| CycleGAN (2017) | 17.678 | 0.615 | 0.392 |
| STGAN (2020) | 18.152 | 0.587 | 0.513 |
| CTGAN (2022) | 18.308 | 0.609 | 0.384 |
| FDH-CR (2026) | 19.477 | 0.682 | 0.299 |
| DAF-Net (2026) | 17.657 | 0.527 | 0.468 |
| CR-former (2024) | 16.434 | 0.541 | 0.468 |
| PMAA (2023) | 18.009 | 0.614 | 0.392 |
| CR-TS Net (2022) | 18.585 | 0.615 | 0.342 |
| UnCRtainTS (2023) | 18.770 | 0.631 | 0.333 |
| TMFNet (2026) | 18.019 | 0.624 | 0.411 |
| DDPM-CR (2023) | 18.742 | 0.614 | 0.329 |
| DiffCR (2024) | 19.150 | 0.531 | 0.255 |
| DE (2024) | 17.210 | 0.483 | 0.349 |
| SRP-CR (2025) | 19.666 | 0.677 | 0.251 |
| PCRDiff (2026) | 19.974 | <u>*0.708*</u> | 0.274 |
| EMRDM (2025) | <u>*20.249*</u> | 0.702 | <u>*0.244*</u> |
| **Ours (SADER)** | **20.941** | **0.729** | **0.228** |
| *提升 (Improvement)* | *+3.42%* | *+2.97%* | *-6.56%* |

</td>
<td valign="top">

| 方法 (Method) | PSNR ↑ | SSIM ↑ | SAM (°) ↓ |
| :--- | :---: | :---: | :---: |
| McGAN (2017) | 25.279 | 0.819 | 10.051 |
| Pix2Pix (2017) | 23.303 | 0.818 | 10.932 |
| CycleGAN (2017) | 26.467 | 0.840 | 12.483 |
| STGAN (2020) | 25.534 | 0.731 | 12.992 |
| PMAA (2023) | 28.396 | 0.771 | 16.776 |
| DAF-Net (2026) | 27.111 | 0.617 | 6.247 |
| CR-former (2024) | 25.845 | 0.603 | 12.166 |
| CR-TS Net (2022) | 26.056 | 0.808 | 11.570 |
| U-TAE (2021) | 26.149 | 0.849 | 10.292 |
| TMFNet (2026) | 23.238 | 0.598 | 7.219 |
| UnCRtainTS (2023) | 28.756 | 0.914 | 8.428 |
| DiffCR (2024) | 26.072 | 0.522 | 11.662 |
| DE (2024) | 26.337 | 0.570 | 10.412 |
| SeqDMs (2023) | 28.074 | 0.827 | 12.777 |
| PCRDiff (2026) | 27.589 | 0.834 | 7.623 |
| EMRDM (2025) | <u>*29.320*</u> | <u>*0.926*</u> | <u>*5.933*</u> |
| **Ours (SADER)** | **31.234** | **0.937** | **5.885** |
| *提升 (Improvement)* | *+6.53%* | *+1.19%* | *-0.81%* |

</td>
</tr>
</table>

## 使用方法 Usage
### 环境配置 Setup
根目录下提供了 requirements.txt 文件，包含我们使用的全部依赖包。但不建议直接运行 pip install -r requirements.txt，因部分包存在复杂依赖关系。建议优先安装核心依赖（如 `torch`、`lightning`）。
推荐使用 conda 创建虚拟环境：
```
conda create --name credm python=3.12
conda activate credm
```
安装 PyTorch (CUDA 12.0)：
```
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
其余依赖请参照 requirements.txt 安装。

### 数据集 Dataset
支持两个数据集： Sen2\_MTC\_New 、 SEN12MS-CR-TS(EA)

> **关于 SEN12MS-CR-TS (东亚子集 EA) 的说明**：在 SEN12MS-CR-TS 东亚子集的评估中，测试集具体采用标准测试区域 `ROIs1868/73`（包含 240 组多时相测试序列样本）进行评测。

### 训练 Train
配置文件位于 ./configs/example_training/：
```
python main.py --base configs/example_training/[yaml_file_name].yaml --enable_tf32
```
可通过 -l 指定日志路径（默认 ./logs）：
```
python main.py --base configs/example_training/[yaml_file_name].yaml --enable_tf32 -l [path_to_your_logs]
```
### 测试 Test
`[yaml_file_name].yaml`与训练过程相同，注意需正确配置 data.params.test 参数：
```
python main.py --base configs/example_training/[yaml_file_name].yaml --enable_tf32 -t false
```

### 预测 Predict
（仅支持单GPU，需设置 lightning.trainer.devices=1）
```
python main.py --base configs/example_training/[yaml_file_name].yaml --enable_tf32 -t false --no-test true --predict true
```
`[yaml_file_name].yaml`与训练和测试过程相同
### 其他 Others
如果有专业问题或者学术合作请联系**YIFAN ZHANG**

联系方式：

**网易邮箱**: ZYFzlblyh20020730@163.com

**微信与手机号**：13796823193

**QQ号**：1740166370
