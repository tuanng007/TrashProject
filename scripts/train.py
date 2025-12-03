"""CLI training script for waste classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.training.dataset import DataConfig
from src.training.losses import LossConfig
from src.training.optim import OptimConfig, SchedulerConfig
from src.training.trainer import TrainConfig, WasteTrainer
from src.utils.seed import DEFAULT_SEED, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train waste classifier locally.")
    parser.add_argument("--train-dir", type=Path, required=True, help="Folder with training images.")
    parser.add_argument("--val-dir", type=Path, required=True, help="Folder with validation images.")
    parser.add_argument("--test-dir", type=Path, default=None, help="Folder with test images (optional).")
    parser.add_argument("--img-size", type=int, default=224, help="Resize image size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Model name: resnet18/mobilenetv3[large]/efficientnetb0/vitb16.",
    )
    parser.add_argument("--loss", type=str, default="cross_entropy", help="Loss type: cross_entropy or focal.")
    parser.add_argument("--use-blur", action="store_true", help="Enable Gaussian blur in train augmentation.")
    parser.add_argument(
        "--use-random-erasing",
        action="store_true",
        help="Enable RandomErasing (Cutout) after normalization.",
    )
    parser.add_argument("--use-mixup", action="store_true", help="Enable Mixup during training.")
    parser.add_argument("--use-cutmix", action="store_true", help="Enable CutMix during training.")
    parser.add_argument("--mixup-alpha", type=float, default=0.4, help="Alpha parameter for Mixup.")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0, help="Alpha parameter for CutMix.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--scheduler", type=str, default="onecycle", help="Scheduler: onecycle/cosine/step/none.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Directory to save checkpoints.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for reproducibility.")
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Force deterministic algorithms (default).",
    )
    parser.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        help="Allow non-deterministic ops (avoid cuBLAS deterministic errors).",
    )
    parser.set_defaults(deterministic=True)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop early if no val acc improvement for this many epochs (0 disables).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum delta in val acc to count as improvement for early stopping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)

    data_cfg = DataConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=4,
        use_blur=args.use_blur,
        use_random_erasing=args.use_random_erasing,
        seed=args.seed,
    )

    loss_cfg = LossConfig(name=args.loss)
    optim_cfg = OptimConfig(lr=args.lr, weight_decay=args.weight_decay)
    scheduler_cfg = SchedulerConfig(name=args.scheduler if args.scheduler != "none" else None)

    train_cfg = TrainConfig(
        data=data_cfg,
        loss=loss_cfg,
        optim=optim_cfg,
        scheduler=scheduler_cfg,
        epochs=args.epochs,
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        use_mixup=args.use_mixup,
        use_cutmix=args.use_cutmix,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        deterministic=args.deterministic,
        early_stop_patience=args.early_stop_patience or None,
        early_stop_min_delta=args.early_stop_min_delta,
    )

    trainer = WasteTrainer(train_cfg)
    report, cm = trainer.train()
    if report:
        print("Classification report:", report.get("macro avg", report))
    if cm is not None:
        print("Confusion matrix saved at:", train_cfg.output_dir / "confusion_matrix.pth")


if __name__ == "__main__":
    main()
