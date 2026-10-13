"""用小型 PIL 数据验证数据边界，离线运行，不下载 CIFAR-10。"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.datasets import create_cifar10_loaders, split_indices
from src.day3_augmentation import save_comparison
from src.transforms import build_transform, to_display_image


class TinyCIFAR10(Dataset):
    """保留真实数据的 PIL/RGB/整数标签接口，训练和测试使用不同来源。"""

    classes = [f"class_{i}" for i in range(10)]

    def __init__(self, root, train=True, download=False, transform=None):
        self.train = train
        self.transform = transform
        size = 37 if train else 11
        rng = np.random.default_rng(7 if train else 8)
        self.data = rng.integers(0, 256, (size, 32, 32, 3), dtype=np.uint8)
        self.targets = [i % 10 for i in range(size)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("src.datasets.CIFAR10", TinyCIFAR10)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_split_is_disjoint_complete_and_reproducible(self):
        train, validation, test = create_cifar10_loaders(download=False)
        train_ids = train.dataset.indices
        validation_ids = validation.dataset.indices
        self.assertFalse(set(train_ids) & set(validation_ids))
        self.assertEqual(set(train_ids) | set(validation_ids), set(range(37)))
        repeated = create_cifar10_loaders(augment=True, download=False)
        self.assertEqual(train_ids, repeated[0].dataset.indices)
        self.assertEqual(validation_ids, repeated[1].dataset.indices)
        self.assertIsNot(train.dataset.dataset, validation.dataset.dataset)
        self.assertTrue(train.dataset.dataset.train)
        self.assertFalse(test.dataset.train)
        self.assertEqual(len(test.dataset), 11)

    def test_validation_and_test_stay_deterministic_with_augmentation(self):
        _, validation, test = create_cifar10_loaders(augment=True, download=False)
        baseline = create_cifar10_loaders(augment=False, download=False)
        for loader, plain in zip((validation, test), baseline[1:]):
            first, label = loader.dataset[0]
            self.assertTrue(torch.equal(first, loader.dataset[0][0]))
            self.assertTrue(torch.equal(first, plain.dataset[0][0]))
            self.assertEqual(label, plain.dataset[0][1])
            first_epoch = torch.cat([images for images, _ in loader])
            second_epoch = torch.cat([images for images, _ in loader])
            self.assertTrue(torch.equal(first_epoch, second_epoch))

    def test_batches_keep_tail_and_valid_shapes(self):
        loaders = create_cifar10_loaders(batch_size=8, augment=True, download=False)
        for loader in loaders:
            count = 0
            for images, labels in loader:
                self.assertEqual(tuple(images.shape[1:]), (3, 32, 32))
                self.assertEqual(images.dtype, torch.float32)
                self.assertTrue(torch.isfinite(images).all())
                self.assertEqual(tuple(labels.shape), (len(images),))
                self.assertEqual(labels.dtype, torch.int64)
                self.assertTrue(((labels >= 0) & (labels < 10)).all())
                count += len(images)
            self.assertEqual(count, len(loader.dataset))

    def test_shuffle_is_reproducible_and_independent_of_validation(self):
        state = torch.random.get_rng_state().clone()
        first = create_cifar10_loaders(download=False)
        second = create_cifar10_loaders(download=False)
        list(first[1])  # 读取验证集不消耗训练集的 generator。
        order = list(iter(first[0].sampler))
        self.assertEqual(order, list(iter(second[0].sampler)))
        self.assertEqual(sorted(order), list(range(len(first[0].dataset))))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_preprocessing_round_trip_preserves_pixels(self):
        source = TinyCIFAR10(root="unused")
        image, _ = source[0]
        tensor = build_transform()(image)
        np.testing.assert_allclose(to_display_image(tensor), source.data[0] / 255,
                                   atol=1e-7)

    def test_invalid_sizes_fail_clearly(self):
        for fraction in (0, 1, -0.1, float("nan")):
            with self.assertRaises(ValueError):
                split_indices(37, validation_fraction=fraction)
        with self.assertRaises(ValueError):
            split_indices(1)
        with self.assertRaises(ValueError):
            create_cifar10_loaders(batch_size=0, download=False)

    @patch("src.day3_augmentation.CIFAR10", TinyCIFAR10)
    def test_preview_saves_single_row_and_restores_rng(self):
        state = torch.random.get_rng_state().clone()
        seen_indices = []
        original_getitem = TinyCIFAR10.__getitem__

        def record_sample(dataset, index):
            seen_indices.append(index)
            return original_getitem(dataset, index)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "comparison.png"
            with patch.object(TinyCIFAR10, "__getitem__", record_sample):
                save_comparison(output=output, samples=1, download=False,
                                validation_fraction=0.2)
            train_indices, _ = split_indices(37, validation_fraction=0.2)
            self.assertEqual(seen_indices, train_indices[:1])
            with Image.open(output) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, image.height)
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
