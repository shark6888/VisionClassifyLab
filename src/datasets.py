"""Day 2：Dataset 取单个样本，DataLoader 把样本组合成 batch。"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from src.transforms import build_transform


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def split_indices(size, validation_fraction=0.1, seed=42):
    """固定索引划分；shuffle 只改变取样顺序，不改变样本所属集合。"""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    validation_size = int(size * validation_fraction)
    if not 0 < validation_size < size:
        raise ValueError("train and validation must each contain at least one sample")
    # 局部 generator 保证同一 seed 的划分可复现，不改变全局随机状态。
    indices = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    return indices[validation_size:], indices[:validation_size]


def create_cifar10_datasets(data_dir=DATA_DIR, validation_fraction=0.1,
                           seed=42, augment=False, download=True):
    """官方 50,000 张训练图划分为 train/validation，10,000 张 test 保持独立。"""
    # 两个实例读取同一份训练文件，但各自拥有 transform。
    # 不要对共享同一 dataset 的两个 Subset 修改 dataset.transform，
    # 否则验证集也会意外得到训练集的随机增强。
    train_source = CIFAR10(root=str(data_dir), train=True, download=download,
                          transform=build_transform(augment=augment))
    validation_source = CIFAR10(root=str(data_dir), train=True, download=False,
                               transform=build_transform())
    train_indices, validation_indices = split_indices(
        len(train_source), validation_fraction, seed
    )
    train_dataset = Subset(train_source, train_indices)
    validation_dataset = Subset(validation_source, validation_indices)
    test_dataset = CIFAR10(root=str(data_dir), train=False, download=download,
                          transform=build_transform())
    return train_dataset, validation_dataset, test_dataset


def create_cifar10_loaders(data_dir=DATA_DIR, batch_size=64, validation_fraction=0.1,
                          seed=42, augment=False, download=True):
    """只有训练集打乱顺序；保留最后一个可能不足 batch_size 的批次。

    seed 固定划分与 shuffle；训练增强使用全局 CPU RNG，
    如需复现增强序列，在实验入口调用 torch.manual_seed(seed)。
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    train, validation, test = create_cifar10_datasets(
        data_dir, validation_fraction, seed, augment, download
    )
    # num_workers=0 便于在 Windows 上直接运行与逐步调试。
    # 每个 loader 使用独立 generator，读取验证集不会改变训练的 shuffle 序列。
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0,
                             generator=torch.Generator().manual_seed(seed))
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False,
                                  num_workers=0, generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(seed))
    return train_loader, validation_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    train_loader, validation_loader, test_loader = create_cifar10_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        validation_fraction=args.validation_fraction, seed=args.seed,
        download=not args.no_download,
    )
    print("Train: update parameters; validation: choose settings; test: final evaluation.")
    print("This demo only inspects data shapes; it does not train or evaluate a model.")
    for name, loader in (("train", train_loader), ("validation", validation_loader),
                         ("test", test_loader)):
        # dataset[i] 返回一张 [3, 32, 32] 图像和一个整数类别编号。
        image, label = loader.dataset[0]
        print(f"{name}: {len(loader.dataset)} samples, {len(loader)} batches")
        print(f"  dataset[0]: image {tuple(image.shape)}, label {label}")
        # DataLoader 堆叠成 images [B, 3, 32, 32]、labels [B]。
        for images, labels in loader:
            print(f"  batch: images {tuple(images.shape)} {images.dtype}, "
                  f"labels {tuple(labels.shape)} {labels.dtype}")
            break
    print("train shuffle=True; validation/test shuffle=False; drop_last=False")


if __name__ == "__main__":
    main()
