# VisionClassifyLab

一个通过约 14 天实践，逐步学习 PyTorch 图像分类的学习型项目。

## Day 1：最小图像分类流程

今天的学习目标：

1. 理解 `image → CNN → logits → CrossEntropyLoss → prediction` 的各个环节。
2. 后续使用 MNIST 跑通一个最小图像分类程序。

当前仅完成项目初始化，尚未实现模型、数据加载和训练程序。

### 核心流程

MNIST 的图像经过张量转换后，单张形状为 `[1, 28, 28]`；一个批次的形状为 `[N, 1, 28, 28]`。标签是 `0` 到 `9` 的整数，批次标签形状为 `[N]`。

CNN 接收图像，输出形状为 `[N, 10]` 的 `logits`。这些值是每个类别的原始分数，不是概率。

`CrossEntropyLoss(logits, labels)` 使用原始分数与正确标签计算损失；不要先对 logits 做 softmax。训练时，通过反向传播和优化器更新模型参数。

预测直接使用 `logits.argmax(dim=1)`，选择分数最大的类别。损失计算和预测都从 logits 出发：损失用于学习，预测用于输出分类结果；预测不需要先计算损失。

```text
image → CNN → logits ─┬→ CrossEntropyLoss(logits, labels) → 反向传播 → 参数更新
                      └→ argmax → prediction
```

### 环境准备

建议在项目根目录创建 Python 虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell 激活方式：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS 激活方式：

```bash
source .venv/bin/activate
```

然后安装 Day 1 依赖：

```bash
python -m pip install -r requirements.txt
```

依赖用途：`torch` 用于模型和训练，`torchvision` 用于 MNIST 数据集及图像转张量，`numpy` 用于基础数组操作，`matplotlib` 用于查看图像和绘图。

## 项目结构

```text
VisionClassifyLab/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
├── notebooks/
├── outputs/
└── notes/
    └── learning_log.md
```

- `src/`：存放后续的 Python 源码，如 Day 1 的最小 MNIST 分类程序。
- `notebooks/`：存放可选的交互实验笔记。目前不安装 Jupyter。
- `outputs/`：存放运行产生的图像、日志和模型文件，内容不纳入 Git。
- `notes/`：记录每日学习目标、理解、实验结果和问题。
- `README.md`：项目介绍、环境准备和学习入口。
- `requirements.txt`：Day 1 使用的四项依赖。
- `.gitignore`：排除虚拟环境、缓存、下载的数据和运行产物。

空目录中的 `.gitkeep` 仅用于让 Git 保留目录，不包含程序逻辑。后续 MNIST 数据可以下载到 `data/`，无需提前创建或提交数据集。

当前范围不包含 Data Augmentation、Transfer Learning、Scheduler、Early Stopping 或 Confusion Matrix。
