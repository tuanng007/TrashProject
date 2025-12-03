"""Lightweight training loop for waste classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torchvision import models

from .dataset import create_dataloaders, DataConfig
from .losses import LossConfig, build_loss
from .optim import OptimConfig, SchedulerConfig, build_optimizer, build_scheduler
from ..utils.metrics import accuracy, compute_classification_report, compute_confusion_matrix
from ..utils.seed import seed_everything


@dataclass
class TrainConfig:
    data: DataConfig
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    epochs: int = 15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 20
    num_classes: int = 6
    output_dir: Path = Path("artifacts")
    model_name: str = "resnet18"
    freeze_backbone_epochs: int = 0
    use_mixup: bool = False
    use_cutmix: bool = False
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 1.0
    deterministic: bool = True


def build_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "mobilenetv3":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == "efficientnetb0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name in {"vitb16", "vit_b16"}:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {name}")


class WasteTrainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        seed_everything(config.data.seed, deterministic=config.deterministic)
        self.device = torch.device(config.device)
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.class_to_idx,
        ) = create_dataloaders(config.data)
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.model = build_model(config.model_name, len(self.class_to_idx)).to(self.device)
        if config.freeze_backbone_epochs > 0:
            # Freeze backbone, giữ lại head để fine-tune
            head_keywords = ("fc", "classifier", "head")
            for name, param in self.model.named_parameters():
                if not any(keyword in name for keyword in head_keywords):
                    param.requires_grad = False
        self.criterion = build_loss(config.loss, class_counts=self._count_classes())
        self.optimizer = build_optimizer(self.model, config.optim)
        scheduler_cfg = None
        if config.scheduler and config.scheduler.name:
            scheduler_cfg = SchedulerConfig(
                name=config.scheduler.name,
                max_lr=config.scheduler.max_lr or config.optim.lr,
                steps_per_epoch=len(self.train_loader),
                epochs=config.epochs,
                gamma=config.scheduler.gamma,
                step_size=config.scheduler.step_size,
            )
        self.scheduler = build_scheduler(self.optimizer, scheduler_cfg) if scheduler_cfg else None
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _count_classes(self):
        counts = [0] * len(self.class_to_idx)
        for _, label in self.train_loader.dataset.samples:
            counts[label] += 1
        return counts

    def _mixup_data(
        self, x: torch.Tensor, y: torch.Tensor, alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if alpha <= 0:
            return x, y, y, 1.0
        lam = torch._sample_dirichlet(torch.tensor([alpha, alpha])).max().item()
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _cutmix_data(
        self, x: torch.Tensor, y: torch.Tensor, alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if alpha <= 0:
            return x, y, y, 1.0
        batch_size, _, h, w = x.size()
        lam = torch._sample_dirichlet(torch.tensor([alpha, alpha])).max().item()
        index = torch.randperm(batch_size, device=x.device)
        cut_rat = (1.0 - lam) ** 0.5
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = torch.randint(w, (1,), device=x.device).item()
        cy = torch.randint(h, (1,), device=x.device).item()

        x1 = max(cx - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, w)
        y1 = max(cy - cut_h // 2, 0)
        y2 = min(cy + cut_h // 2, h)

        x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam

    def _compute_loss(
        self,
        outputs: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        if lam == 1.0:
            return self.criterion(outputs, y_a)
        return lam * self.criterion(outputs, y_a) + (1.0 - lam) * self.criterion(outputs, y_b)

    def train(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        best_acc = 0.0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        global_step = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            running_loss = 0.0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                use_mixup = self.config.use_mixup and not self.config.use_cutmix
                use_cutmix = self.config.use_cutmix
                if use_mixup:
                    images, labels_a, labels_b, lam = self._mixup_data(
                        images, labels, self.config.mixup_alpha
                    )
                elif use_cutmix:
                    images, labels_a, labels_b, lam = self._cutmix_data(
                        images, labels, self.config.cutmix_alpha
                    )
                else:
                    labels_a, labels_b, lam = labels, labels, 1.0
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self._compute_loss(outputs, labels_a, labels_b, lam)
                loss.backward()
                self.optimizer.step()
                if self.scheduler and hasattr(self.scheduler, "step") and self.scheduler.__class__.__name__ == "OneCycleLR":
                    self.scheduler.step()
                running_loss += loss.item()
                global_step += 1
                if batch_idx % self.config.log_every == 0:
                    print(
                        f"Epoch {epoch+1}/{self.config.epochs} "
                        f"Step {batch_idx}/{len(self.train_loader)} "
                        f"Loss: {loss.item():.4f}"
                    )

            if self.scheduler and self.scheduler.__class__.__name__ != "OneCycleLR":
                self.scheduler.step()

            avg_train_loss = running_loss / len(self.train_loader)
            val_loss, val_acc = self.evaluate(split="val")
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                self._save_checkpoint(epoch, best=True)
            self._save_checkpoint(epoch, best=False)

            if epoch + 1 == self.config.freeze_backbone_epochs:
                for param in self.model.parameters():
                    param.requires_grad = True

        report, cm = {}, None
        if self.test_loader is not None:
            report, cm = self.test()
        torch.save(history, self.config.output_dir / "history.pth")
        return report, cm

    def evaluate(self, split: str = "val") -> Tuple[float, float]:
        self.model.eval()
        loader = self.val_loader if split == "val" else self.test_loader
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        return total_loss / len(loader), correct / total

    def test(self) -> Tuple[Dict[str, Dict[str, float]], torch.Tensor]:
        if self.test_loader is None:
            raise RuntimeError("Test loader not provided.")
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for images, batch_labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                preds.extend(outputs.argmax(dim=1).cpu().tolist())
                labels.extend(batch_labels.tolist())
        report = compute_classification_report(
            labels,
            preds,
            target_names=[self.idx_to_class[idx] for idx in sorted(self.idx_to_class)],
        )
        cm = compute_confusion_matrix(labels, preds)
        torch.save(report, self.config.output_dir / "classification_report.pth")
        torch.save(cm, self.config.output_dir / "confusion_matrix.pth")
        return report, cm

    def _save_checkpoint(self, epoch: int, best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        path = self.config.output_dir / ("best.pt" if best else f"epoch_{epoch+1}.pt")
        torch.save(state, path)
