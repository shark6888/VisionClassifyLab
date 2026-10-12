"""最小 MNIST CNN 分类实验：观察图像、logits、损失与预测的关系。"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main():
    batch_size = 64
    num_epochs = 1
    data_dir = Path(__file__).resolve().parent.parent / "data"

    # MNIST 原图是 28 × 28 的灰度图。
    # ToTensor 将像素转换为 [0, 1] 的 float32，单张形状为 [1, 28, 28]。
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )

    # DataLoader 把单张图像堆成批次：images 为 [N, 1, 28, 28]。
    # labels 为 [N] 的 int64 张量，每个元素是 0～9 的类别编号。
    # N 是当前 batch 的实际大小，最后一个 batch 可能小于 batch_size。
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),  # [N, 1, 28, 28] → [N, 8, 28, 28]
        nn.ReLU(),  # 形状不变
        nn.MaxPool2d(kernel_size=2),  # [N, 8, 28, 28] → [N, 8, 14, 14]
        nn.Conv2d(8, 16, kernel_size=3, padding=1),  # → [N, 16, 14, 14]
        nn.ReLU(),  # 形状不变
        nn.MaxPool2d(kernel_size=2),  # → [N, 16, 7, 7]
        nn.Flatten(),  # 保留 batch 维度：[N, 16, 7, 7] → [N, 784]
        nn.Linear(16 * 7 * 7, 10),  # [N, 784] → [N, 10]，每类一个原始分数
    )

    # CrossEntropyLoss 接收 logits [N, 10] 和 labels [N]，返回一个标量。
    # logits 是原始分数；这里不先做 softmax，也不将 labels 转成 one-hot。
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_images = 0

        for batch_index, (images, labels) in enumerate(train_loader):
            # 清除上一批梯度 → 前向计算 → 计算损失 → 反向传播 → 更新参数。
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if epoch == 0 and batch_index == 0:
                print("images.shape:", images.shape)
                print("labels.shape:", labels.shape)
                print("logits.shape:", logits.shape)

            # loss 是当前 batch 的平均损失，按实际样本数累计整个 epoch 的均值。
            total_loss += loss.item() * images.size(0)
            total_images += images.size(0)

        print(f"Epoch {epoch + 1}/{num_epochs}, loss: {total_loss / total_images:.4f}")

    model.eval()
    correct = 0
    total = 0

    # 测试时只做前向计算与预测，不计算梯度，也不更新参数。
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)  # [N, 10]
            # dim=1 是类别维度，取最高分的类别后得到 predictions [N]。
            # argmax 可以直接用于 logits，无需先计算损失或 softmax。
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f"Test accuracy: {correct / total:.2%}")


if __name__ == "__main__":
    main()
