"""Dataset and dataloaders for waste classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class WasteDataset(Dataset):
    """Simple dataset that expects folder structure root/class_name/*.jpg."""

    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        if class_to_idx is None:
            classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
            self.class_to_idx = {name: idx for idx, name in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        self.samples = self._collect_samples()

    def _collect_samples(self):
        items = []
        for class_name, idx in self.class_to_idx.items():
            class_dir = self.root / class_name
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    items.append((img_path, idx))
        if not items:
            raise RuntimeError(f"No images found under {self.root}")
        return items

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


@dataclass
class DataConfig:
    train_dir: Path
    val_dir: Path
    test_dir: Optional[Path] = None
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 2


def default_transforms(img_size: int) -> Tuple[Callable, Callable, Callable]:
    """Return train/val/test transforms."""
    train_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.2)),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf, eval_tf


def create_dataloaders(
    config: DataConfig,
    transforms_fn: Callable[[int], Tuple[Callable, Callable, Callable]] = default_transforms,
):
    """Create train/val/test dataloaders along with class mapping."""
    train_tf, val_tf, test_tf = transforms_fn(config.img_size)
    train_dataset = WasteDataset(config.train_dir, transform=train_tf)
    val_dataset = WasteDataset(
        config.val_dir, transform=val_tf, class_to_idx=train_dataset.class_to_idx
    )
    test_loader = None
    if config.test_dir and Path(config.test_dir).exists():
        test_dataset = WasteDataset(
            config.test_dir, transform=test_tf, class_to_idx=train_dataset.class_to_idx
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx

