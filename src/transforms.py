"""Day 3：先统一输入，再为训练图片制造合理变化。"""

import torch
from torchvision import transforms


# 固定缩放：[0, 1] → [-1, 1]，不是从验证集或测试集估计的统计量。
# Normalize 按通道计算 (pixel - mean) / std，不会把图片自动变成正态分布。
CIFAR10_MEAN = (0.5, 0.5, 0.5)
CIFAR10_STD = (0.5, 0.5, 0.5)


def build_transform(augment=False):
    """augment=True 只用于训练集；默认提供确定性的必要预处理。"""
    # CIFAR-10 原生就是 32 × 32；这里保留 Resize 是为了展示尺寸统一步骤。
    steps = [transforms.Resize((32, 32))]
    if augment:
        steps.extend([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10, fill=(128, 128, 128)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ])
    steps.extend([
        # PIL RGB 图像 [H, W, C] → float32 张量 [C, H, W]，像素值 [0, 1]。
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return transforms.Compose(steps)


def to_display_image(image):
    """逆归一化 [C, H, W] → [H, W, C]，用于 imshow，避免显示颜色失真。"""
    image = image.detach().cpu()
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
