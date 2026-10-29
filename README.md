# SAM-LoRA 遥感图像篡改定位

基于 Segment Anything Model（SAM ViT-B）与 LoRA 的像素级遥感图像篡改定位代码，支持复制-移动（copy-move）、拼接（splicing）和移除（remove）三类篡改区域的分割。

在 SAM 图像编码器的注意力 Q/V 分支中加入 LoRA，使用整图边界框 `[0, 0, W, H]` 作为提示，并通过多类别分割头输出背景及三类篡改区域的预测。仓库提供训练、测试集评估和单张图像推理代码。


## 数据与模型下载

代码仓库不包含数据集和 SAM 预训练权重，请通过下方链接单独下载。本项目不发布训练后的模型权重，评估和推理所需参数需由使用者自行训练生成。

| 资源 | 下载地址 | 使用方式 |
| --- | --- | --- |
| 本文数据集（含训练/验证/测试划分及标注） | [RS-TID](https://pan.baidu.com/s/1yY7O23Zs7wBUv_7pYkozng?pwd=4rps ) | 解压为 `all_dataset/`，或修改 `config.yaml` 中的数据路径 |
| SAM ViT-B 官方预训练权重 | [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | 放在项目根目录，或修改 `SAM.CHECKPOINT` |

SAM 官方预训练权重用于初始化模型。请先按下文训练，生成 LoRA 和分类头参数及对应的 checkpoint，再执行评估或推理。

SAM 权重来源及其他模型版本见 [SAM 官方仓库](https://github.com/facebookresearch/segment-anything#model-checkpoints)。

## 目录结构

```text
.
├── README.md
├── .gitignore
├── config.yaml                    # 数据路径、SAM 权重路径与训练超参数
├── train.py                       # 训练
├── evaluate_model_paper.py         # 数据集评估
├── test.py                        # 单张图像推理
├── src/
│   ├── segment_anything/          # 项目使用的 SAM 源码
│   ├── lora.py                    # Q/V 分支 LoRA 与参数存取
│   ├── multiclass_head.py         # 多类别分割头
│   ├── dataloader.py               # 数据集加载
│   ├── processor.py               # 图像与提示预处理
│   └── utils.py                   # 掩码转换、指标及可视化
├── sam_vit_b_01ec64.pth            # 单独下载
├── all_dataset/                   # 单独下载
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
└── training_results/              # 自行训练生成
    └── train_YYYYMMDD_HHMMSS/
        ├── config.yaml
        ├── history.json
        ├── summary.json
        ├── lora_rank512.safetensors
        ├── multiclass_head.pth
        ├── checkpoints/
        │   └── best_model.pth
        └── evaluation_paper/      # 评估后生成
```

## 环境准备

在项目根目录执行以下命令。根据运行环境安装相互兼容的 PyTorch、TorchVision 和 CUDA 版本：

```bash
pip install torch torchvision
pip install numpy pillow pyyaml safetensors tqdm matplotlib seaborn monai
```

仓库已包含 `src/segment_anything/`，主实验无需额外安装 `segment-anything` 包。当前仓库未锁定论文实验环境的具体依赖版本；正式发布时应补充实际使用的 Python、PyTorch、CUDA 版本和 GPU 型号。

下载 SAM ViT-B 权重：

```bash
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o sam_vit_b_01ec64.pth
```

**训练设备配置：** 当前 `train.py` 中分类头和训练设备均指定为 `cuda:1`（第二张可见 GPU）。如果仅有一张可见 GPU，请将脚本中的两处 `cuda:1` 改为 `cuda:0`。评估和单图推理默认使用 `cuda`，无可用 CUDA 时使用 CPU。

## 数据集准备

从上方数据集链接下载并解压，保持 `train/val/test` 划分，各子集分别包含 `images/` 和 `masks/`。使用发布的数据划分进行复现。

- 图像与掩码按不含扩展名的文件名配对，例如 `images/P1726.png` 对应 `masks/P1726.png`。
- 加载器支持 JPG、JPEG 和 PNG；彩色标注建议使用无损 PNG，以保留准确的类别颜色。
- 掩码为 RGB 彩色图，类别映射由 `src/utils.py` 定义。

| 类别 ID | 名称 | RGB 颜色 |
| --- | --- | --- |
| 0 | 未篡改 / 背景 | 黑色 `(0, 0, 0)` |
| 1 | copy_move（复制-移动） | 绿色 `(0, 255, 0)` |
| 2 | splicing（拼接） | 红色 `(255, 0, 0)` |
| 3 | remove（移除） | 蓝色 `(0, 0, 255)` |

现有转换函数也将白色背景 `(255, 255, 255)` 解析为类别 0。

## 配置

修改 `config.yaml`，使数据与权重路径指向本地文件。当前默认配置为：

```yaml
DATASET:
  TEST_PATH: ./all_dataset/test
  TRAIN_PATH: ./all_dataset/train
  VAL_PATH: ./all_dataset/val
SAM:
  CHECKPOINT: ./sam_vit_b_01ec64.pth
  RANK: 512
TRAIN:
  BATCH_SIZE: 1
  GRADIENT_ACCUMULATION_STEPS: 16
  LEARNING_RATE: 1e-4
  NUM_EPOCHS: 50
  WEIGHT_DECAY: 0
```

默认有效 batch size 为 `1 × 16 = 16`。训练默认随机种子为 42，可在 `TRAIN` 中添加 `SEED` 修改；早停耐心值等参数在 `train.py` 顶部设置。

## 训练

完成数据集、SAM 权重和设备配置后运行：

```bash
python train.py
```

训练优化图像编码器中的 LoRA 参数及多类别分割头，使用 Dice 与带类别权重的交叉熵组合损失。分割头使用两层 3×3 卷积、GroupNorm、GELU、Dropout 和残差分支。

结果保存至 `training_results/train_YYYYMMDD_HHMMSS/`：

| 文件 | 内容 |
| --- | --- |
| `config.yaml` | 本次训练配置 |
| `history.json` / `summary.json` | 训练记录与摘要 |
| `lora_rank{RANK}.safetensors` | 训练结束时导出的 LoRA 参数 |
| `multiclass_head.pth` | 训练结束时导出的分类头参数 |
| `checkpoints/best_model.pth` | 验证损失最优时的完整 SAM（含 LoRA）、分类头与优化器状态 |

**训练结束导出的两个参数文件与验证集最优 checkpoint 可能来自不同 epoch。** 报告结果时，应说明使用的 epoch 和加载方式。

## 评估

修改 `evaluate_model_paper.py` 顶部的路径：

```python
TRAINING_RESULT_DIR = "./training_results/train_YYYYMMDD_HHMMSS"
```

确认 `config.yaml` 中的 SAM 权重、LoRA rank 与数据路径匹配该次实验，然后运行：

```bash
python evaluate_model_paper.py
```

默认 `USE_TEST_SET = True`，在测试集上评估。`LOAD_FROM_BEST_CHECKPOINT = True` 时优先加载 `checkpoints/best_model.pth`；若该文件不存在，则加载同目录中的 `lora_rank{RANK}.safetensors` 与 `multiclass_head.pth`。

指标基于整个评估集累计的像素混淆矩阵计算，预测掩码恢复至真实标注尺寸后进行比较，同时报告含背景四类和仅篡改三类的宏平均指标。输出位于训练目录下的 `evaluation_paper/`：

- `paper_summary.json`：总体准确率及宏平均 Dice、IoU、F1。
- `metrics_paper_ready.json`：逐类指标及汇总指标。
- `confusion_matrix_raw.json`：原始混淆矩阵。
- `confusion_matrix_row_normalized.json`：行归一化混淆矩阵。
- `confusion_matrix.png`：混淆矩阵图。

论文结果请以对应训练权重、数据划分和评估脚本实际生成的文件为准。

## 单张图像推理

修改 `test.py` 顶部的配置：

```python
IMAGE_PATH = "./all_dataset/test/images/P1726.png"
CONFIG_FILE = "./config.yaml"
SAM_CHECKPOINT = "./sam_vit_b_01ec64.pth"
TRAINING_RESULT_DIR = "./training_results/train_YYYYMMDD_HHMMSS"
OUTPUT_DIR = "./test_results/single_image"
```

```bash
python test.py
```

输出包含彩色预测掩码与可视化图。单图推理主要加载训练目录中的 `lora_rank{RANK}.safetensors` 和 `multiclass_head.pth`；请确保两者来自同一次训练、同一 epoch，并与配置中的 rank 一致。该脚本不会像评估脚本一样直接恢复完整最优 checkpoint。

## 致谢

本项目基于以下开源工作改造：

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Sam_LoRA — JamesQFreeman](https://github.com/JamesQFreeman/Sam_LoRA)
- [Segment Your Ring / Sam_LoRA — MathieuNlp](https://github.com/MathieuNlp/Sam_LoRA)
- [Fine-tune SAM tutorial — Niels Rogge](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb)
