"""Day 3：同一张原图的 augmentation OFF / ON 对照，不进行模型训练。"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 保存 PNG 即可，无图形桌面或远程终端也能运行。
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import CIFAR10

from src.datasets import DATA_DIR, split_indices
from src.transforms import build_transform, to_display_image


OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "day3_augmentation.png"


def save_comparison(data_dir=DATA_DIR, output=OUTPUT_PATH, samples=4,
                    seed=42, download=True, validation_fraction=0.1):
    if not 1 <= samples <= 10:
        raise ValueError("samples must be between 1 and 10")
    # 读取未变换的 PIL 图片；使用与 Day 2 一致的比例和 seed 选取训练示例。
    dataset = CIFAR10(root=str(data_dir), train=True, download=download)
    train_indices, _ = split_indices(len(dataset), validation_fraction, seed)
    if samples > len(train_indices):
        raise ValueError("samples exceeds the training split size")
    preprocessing = build_transform(augment=False)
    augmentation = build_transform(augment=True)
    fig, axes = plt.subplots(samples, 5, figsize=(12, 2.5 * samples), squeeze=False)
    try:
        # 可复现预览，并在结束时恢复全局 CPU RNG，不影响调用方之后的实验。
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            for row, index in enumerate(train_indices[:samples]):
                original, label = dataset[index]
                axes[row, 0].imshow(original)
                axes[row, 0].set_title(f"Original #{index}\n{dataset.classes[label]}")
                axes[row, 1].imshow(to_display_image(preprocessing(original)))
                axes[row, 1].set_title("OFF: preprocess only")
                # 同一张原图反复变换，标签保持不变；不是比较不同图片。
                for col in range(2, 5):
                    axes[row, col].imshow(to_display_image(augmentation(original)))
                    axes[row, col].set_title(f"ON: sample {col - 1}")
                for axis in axes[row]:
                    axis.axis("off")
        fig.tight_layout()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=140)
    finally:
        plt.close(fig)
    print(f"Saved comparison: {output.resolve()}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    save_comparison(data_dir=args.data_dir, output=args.output, samples=args.samples,
                    seed=args.seed, download=not args.no_download,
                    validation_fraction=args.validation_fraction)


if __name__ == "__main__":
    main()
