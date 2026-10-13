# 学习日志

## Day 1：最小图像分类流程

### 学习目标

- 理解 image → CNN → logits → CrossEntropyLoss → prediction。
- 使用 MNIST 跑通一个最小 PyTorch 图像分类程序。

### 当前进度

- 已初始化项目目录和依赖清单。
- 已实现最小 MNIST CNN 分类程序，默认训练 1 个 epoch。
- 尚未实际运行程序，实验结果待手动验证。

### 关键理解

- 输入：MNIST 是单通道的 28 × 28 图像，批次张量形状为 `[N, 1, 28, 28]`。
- 输出：模型为每张图像输出 10 个 logits，形状为 `[N, 10]`。
- 损失：CrossEntropyLoss 接收原始 logits 和整数类别标签，不需要提前 softmax。
- 预测：沿类别维度取 argmax，得到 `0` 到 `9` 的预测类别。
- 训练：计算损失后反向传播，通过优化器更新模型参数。

### 实验记录（完成运行后填写）

- 运行命令：
- 训练损失：
- 测试准确率：
- 观察与结论：

### 待思考

- logits 和概率有什么区别？
- 为什么 CrossEntropyLoss 前不需要 softmax？
- 为什么 prediction 可以直接从 logits 得到？

## Day 2：Dataset、DataLoader 与数据划分

### 学习目标

- 从一张图片和一个标签，理解如何组成训练所需的 batch。
- 区分用于更新参数的 train、用于选择设置的 validation 和用于最终评估的 test。

### 当前进度

- 已实现 `src/datasets.py`，提供 CIFAR-10 数据集及 DataLoader 的创建函数。
- 默认从官方训练集划分 45,000 张 train、5,000 张 validation；官方 10,000 张 test 独立保留。
- 固定 seed 可复现划分与 shuffle；训练／验证索引互斥，验证／测试不打乱顺序。
- 可运行 `python -m src.datasets`，观察单张 `[3, 32, 32]` 和批次 `[B, 3, 32, 32]`。

### 我需要能解释什么（自测）

- `dataset[i]` 与 `for images, labels in loader` 分别返回什么？
- 为什么 shuffle 不等于重新划分 train／validation？
- 为什么不能用测试结果选择参数？
- batch size 改为 32 后，张量形状和 batch 数如何变化？

### 还不理解什么（学习后填写）

- 待填写。

## Day 3：预处理与数据增强

### 学习目标

- 预处理统一输入的尺寸、类型与尺度；增强制造标签仍成立的训练变化。
- 明白验证／测试必须使用稳定的必要预处理，不能混入随机增强。

### 当前进度

- 已实现 Resize、ToTensor、Normalize，以及训练可选的翻转、裁剪、轻微旋转、颜色变化。
- Normalize 使用固定的通道 mean/std = 0.5，把 `[0, 1]` 映射到 `[-1, 1]`，没有使用验证／测试统计量。
- 已实现 `python -m src.day3_augmentation`，输出 `outputs/day3_augmentation.png`。
- 每行展示同一原图、OFF 和三次 ON；绘图前逆归一化，便于观察真实颜色。
- Day 2／3 要沿用相同的 seed 和 validation fraction，才能保持同一训练划分。

### 我需要能解释什么（自测）

- ToTensor 和 Normalize 分别改变什么？为什么 Normalize 不等于正态分布？
- 为什么不能对两个共享同一 dataset 的 Subset 分别修改 `dataset.transform`？
- 为什么原图和 OFF 应基本一致，而 ON 可能每次不同？
- 哪些增强会破坏类别含义？应该用哪个集合判断增强是否有效？
- 为什么 loader 的 seed 不等于控制了全部随机增强？

### 还不理解什么（学习后填写）

- 待填写。

## Day 2／3 工程验证（2026-10-13）

- 环境：项目 `.venv`，Python 3.12、PyTorch 2.14.1+cpu、TorchVision 0.29.1+cpu；已补齐原依赖清单中的 Matplotlib。
- `python -m unittest discover -s tests -v`：7 项测试通过，使用合成 PIL 图片，无需网络。
- 覆盖：划分完整且无交集、固定划分、评估预处理稳定、正确 batch 形状及尾批、shuffle 与评估随机状态独立、逆归一化、非默认划分下单行预览保存及 RNG 恢复。
- 两个命令入口的 `--help` 和 `git diff --check` 通过。
- 官方 CIFAR-10 下载出现远端连接中断；真实数据入口和真实图片对照尚未验证。这里没有记录或推测分类准确率。
- 网络恢复后，先运行 `python -m src.datasets`，再运行 `python -m src.day3_augmentation --no-download`，填写图片观察结果。
- 本阶段仅实现数据部分；CIFAR-10 训练循环、选模型和分类指标留给后续学习日。
