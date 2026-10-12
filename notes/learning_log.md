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
