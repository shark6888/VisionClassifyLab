# VisionClassifyLab

一个使用 PyTorch 学习计算机视觉图像分类的实验项目。从小规模、可直接阅读的程序开始，逐步理解数据加载、模型前向计算、损失、反向传播、参数更新和预测。

项目以理解完整分类流程为目标，优先保持代码简单，并记录实验中的张量形状、结果和疑问。目前实现到 Day 3。

## 项目结构

```text
VisionClassifyLab/
├── AGENTS.md                 # 项目规则与 Git 提交规范
├── README.md                 # 项目介绍与运行入口
├── requirements.txt          # Python 依赖
├── .gitignore                # Git 忽略规则
├── src/
│   ├── day1_mnist.py          # Day 1：最小 MNIST CNN 分类实验
│   ├── datasets.py            # Day 2：CIFAR-10 划分、Dataset 与 DataLoader
│   ├── transforms.py          # Day 3：预处理、训练增强与显示用逆归一化
│   └── day3_augmentation.py   # Day 3：同图 OFF / ON 对照
├── tests/
│   └── test_data_pipeline.py # 无需下载数据的行为测试
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
- `torchvision`：加载图像数据集、预处理与数据增强。
- `numpy`：基础数组操作。
- `matplotlib`：查看图像和绘图。

以下命令均在项目根目录、已激活的虚拟环境中运行。Day 2／3 只处理数据，CPU 即可；没有新增第三方依赖。

## Day 1：最小分类流程

使用 MNIST 和一个简单 CNN，默认训练 1 个 epoch：

```bash
python src/day1_mnist.py
```

首次运行需要联网下载 MNIST 到项目的 `data/` 目录。程序会打印第一个训练 batch 的图像、标签和 logits 形状，随后输出平均训练损失和测试准确率。

```text
MNIST → DataLoader → CNN → logits
                           ├→ CrossEntropyLoss(logits, labels) → backward → optimizer.step()
                           └→ argmax → prediction
```

每批图像形状为 `[N, C, H, W]`，模型输出为 `[N, 类别数]` 的原始分数。`CrossEntropyLoss` 使用这些分数和整数类别标签计算损失；预测沿类别维度取 `argmax`。实验源码中的注释给出每层张量形状。

## Day 2：样本、批次与数据划分

从最小需求出发：模型每次需要一批图片和正确标签，而判断泛化能力需要没有参与参数更新的数据。

```bash
python -m src.datasets
python -m src.datasets --batch-size 32 --seed 42 --no-download
```

首次运行下载官方 CIFAR-10（压缩包约 170 MB）；以后使用缓存。`--no-download` 用于明确禁止下载，缓存缺失时会报错。默认 `data/` 路径与当前启动目录无关，也可通过 `--data-dir` 指定。

| 集合 | 默认数量 | 用途 | shuffle | 随机增强 |
|---|---:|---|---|---|
| train | 45,000 | 更新模型参数 | True | Day 3 可开启 |
| validation | 5,000 | 比较设置、选择模型 | False | 无 |
| test | 10,000 | 设置确定后的最终评估 | False | 无 |

训练／验证索引仅从官方 50,000 张训练图中划分；官方测试集独立保留。默认验证比例为 `0.1`，可通过 `--validation-fraction` 调整。相同 seed 固定划分，`shuffle` 只改变训练样本的取出顺序，**不会重新划分数据集**。当前脚本只检查数据形状，不计算测试准确率，也不调参。

阅读 `src/datasets.py` 中的两处核心语句：

- `image, label = loader.dataset[0]`：一张图像 `[3, 32, 32]` 和一个整数类别编号。
- `for images, labels in loader`：一批图像 `[B, 3, 32, 32]` 和标签 `[B]`；图像为 `float32`，标签为 `int64`。

把 batch size 从 64 改为 32，观察形状与 batch 数。最后一批可能不足设定大小，仍会保留。验证集可用于后续选模型，但不能用于反向传播；测试集不能用于选择增强策略或其他参数。

## Day 3：预处理与训练增强

预处理让输入尺寸、类型与数值尺度一致；增强在标签仍成立的前提下，制造训练时可见的合理变化。是否改善泛化，需要后续对比验证集结果才能判断。

| 操作 | 解决的问题 |
|---|---|
| `Resize((32, 32))` | 统一空间尺寸；CIFAR-10 原生已是 32×32，此步不改变分辨率 |
| `ToTensor` | RGB 像素 `[H, W, C]` 转为 `[C, H, W]` 浮点张量，数值从 0～255 缩放到 0～1 |
| `Normalize` | 按通道计算 `(x - mean) / std`；本实验固定 mean/std 为 0.5，得到 -1～1，不代表强制正态分布 |
| `RandomCrop`、`RandomHorizontalFlip` | 小幅改变位置与左右朝向 |
| `RandomRotation`、`ColorJitter` | 轻微旋转与颜色变化 |

训练流水线为 `Resize → 可选随机增强 → ToTensor → Normalize`。验证／测试始终只做 `Resize → ToTensor → Normalize`，确保重复读取结果稳定。训练和验证虽然读取同一份原始文件，但使用独立 Dataset 实例，避免修改共享 transform 造成泄漏。

生成可复现的图片对照：

```bash
python -m src.day3_augmentation
python -m src.day3_augmentation --samples 1 --seed 7 --no-download --output outputs/day3_seed7.png
```

默认输出 `outputs/day3_augmentation.png`。每行是同一张训练图片：原图、OFF（仅预处理）、三次 ON（随机增强）。显示前逆归一化，所以原图和 OFF 的颜色应一致；ON 列展示随机变化，可能偶然相似。脚本保存图片，不弹出窗口。

如果 Day 2 修改了 `--seed` 或 `--validation-fraction`，Day 3 也传入相同值，才能沿用同一训练／验证划分。

后续训练代码可以直接复用开关：

```python
import torch
from src.datasets import create_cifar10_loaders

torch.manual_seed(42)  # 在实验入口固定训练增强所使用的全局随机状态。
train_loader, validation_loader, test_loader = create_cifar10_loaders(augment=True)
```

loader 的 `seed` 固定数据划分和训练 shuffle；训练增强使用全局 CPU 随机状态，需要在实验入口单独设种子。预览脚本则自行固定并恢复随机状态。它们不承诺未来完整训练过程完全确定。当前采用 `num_workers=0`，便于 Windows 运行和逐步调试。

观察时思考：变化后标签是否还合理？颜色变化是否过强？如果验证图片也随机变化，比较两次模型结果还公平吗？Day 2／3 到此为止，CIFAR-10 训练循环留给 Day 4。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试使用小型合成 PIL 图片，不下载数据。覆盖划分不重叠且完整、验证／测试稳定、shuffle 随机状态独立、batch 形状与尾批保留、逆归一化还原像素及单行预览保存。

## Git 提交规范

提交信息采用 `type(scope): description` 格式，例如 `feat(mnist): add minimal CNN classification experiment`。具体约定见 `AGENTS.md`。
