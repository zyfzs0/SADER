
# SADER模型
本仓库为**SADER**模型的官方实现，该方法通过通过引入全新的**MTCDN**网络和基于云层注意力损失函数和重采样，在遥感图像去云任务中展现出卓越性能。
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
支持两个数据集： Sen2\_MTC\_New 、 TS
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
如果有专业问题或者学术合作请联系**张一凡同学**

联系方式：

**网易邮箱**: ZYFzlblyh20020730@163.com

**微信与手机号**：13796823193

**QQ号**：1740166370
