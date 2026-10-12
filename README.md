# VisionClassifyLab

一个使用 PyTorch 学习计算机视觉图像分类的实验项目。从小规模、可直接阅读的程序开始，逐步理解数据加载、模型前向计算、损失、反向传播、参数更新和预测。

项目以理解完整分类流程为目标，优先保持代码简单，并记录实验中的张量形状、结果和疑问。

## 项目结构

```text
VisionClassifyLab/
├── AGENTS.md                 # 项目规则与 Git 提交规范
├── README.md                 # 项目介绍与运行入口
├── requirements.txt          # Python 依赖
├── .gitignore                # Git 忽略规则
├── src/                      # 可直接运行的实验源码
│   └── day1_mnist.py          # 最小 MNIST CNN 分类实验
├── notebooks/                # 可选的交互实验笔记
├── outputs/                  # 实验运行产物
└── notes/
    └── learning_log.md       # 学习记录、实验结果和问题
```

`data/` 在运行时用于保存下载的数据集；数据集、虚拟环境、缓存和 `outputs/` 中的运行产物不提交到 Git。空目录中的 `.gitkeep` 用于让 Git 保留目录。

## 环境准备

在项目根目录创建 Python 虚拟环境：

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

安装依赖：

```bash
python -m pip install -r requirements.txt
```

- `torch`：构建模型、计算损失和训练。
- `torchvision`：加载图像数据集及将图像转换为张量。
- `numpy`：基础数组操作。
- `matplotlib`：查看图像和绘图。

## 运行实验

当前实验使用 MNIST 和一个简单 CNN，默认训练 1 个 epoch：

```bash
python src/day1_mnist.py
```

首次运行需要联网下载 MNIST 到项目的 `data/` 目录。程序会打印第一个训练 batch 的图像、标签和 logits 形状，随后输出平均训练损失和测试准确率。

## 分类流程

```text
MNIST → DataLoader → CNN → logits
                           ├→ CrossEntropyLoss(logits, labels) → backward → optimizer.step()
                           └→ argmax → prediction
```

每批图像形状为 `[N, C, H, W]`，模型输出为 `[N, 类别数]` 的原始分数。`CrossEntropyLoss` 使用这些分数和整数类别标签计算损失；预测沿类别维度取 `argmax`。实验源码中的注释给出每层张量形状，学习过程记录在 `notes/learning_log.md`。

## Git 提交规范

提交信息采用 `type(scope): description` 格式，例如 `feat(mnist): add minimal CNN classification experiment`。具体约定见 `AGENTS.md`。
