"""Gradio demo app for the waste classification model.

Chức năng:
- Load checkpoint tốt nhất (mặc định: artifacts/best.pt).
- Sử dụng lại kiến trúc (ResNet18/MobileNetV3/EfficientNetB0) và preprocessing
  giống pipeline huấn luyện.
- Cho phép người dùng upload ảnh và xem top-k lớp dự đoán cùng xác suất.

Ví dụ chạy:
    python -m scripts.demo_gradio --train-dir data/train --checkpoint artifacts/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
from PIL import Image
import torch

from src.training.dataset import WasteDataset, default_transforms
from src.training.trainer import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio demo for waste classifier.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="Thư mục train (dùng để suy ra danh sách lớp).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/best.pt"),
        help="Đường dẫn tới checkpoint đã huấn luyện.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Tên mô hình (resnet18/mobilenetv3/efficientnetb0).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Kích thước ảnh đầu vào (phải khớp với lúc huấn luyện).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị suy luận (cuda/cpu).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Bật share=True để Gradio tạo link public tạm thời.",
    )
    return parser.parse_args()


def load_artifacts(
    train_dir: Path,
    checkpoint: Path,
    model_name: str,
    img_size: int,
    device: str,
) -> Tuple[torch.nn.Module, callable, Dict[int, str]]:
    # Suy ra mapping lớp từ thư mục train
    dataset = WasteDataset(train_dir)
    idx_to_class = {idx: cls for cls, idx in dataset.class_to_idx.items()}
    num_classes = len(idx_to_class)

    # Build model và load weights
    model = build_model(model_name, num_classes)
    state = torch.load(checkpoint, map_location=device)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        # Trường hợp checkpoint chỉ chứa state_dict thuần
        model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Dùng eval transform giống pipeline huấn luyện
    _, eval_tf, _ = default_transforms(img_size)

    return model, eval_tf, idx_to_class


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model, eval_tf, idx_to_class = load_artifacts(
        train_dir=args.train_dir,
        checkpoint=args.checkpoint,
        model_name=args.model,
        img_size=args.img_size,
        device=device,
    )

    def predict(image: Image.Image):
        if image is None:
            return {}
        x = eval_tf(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu()

        # Trả về dict {class_name: probability}
        scores = {idx_to_class[idx]: float(probs[idx]) for idx in range(len(idx_to_class))}
        # Gradio Label sẽ tự sắp xếp theo xác suất
        return scores

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload ảnh rác"),
        outputs=gr.Label(num_top_classes=3, label="Dự đoán loại rác"),
        title="Waste Classification Demo",
        description=(
            "Upload một ảnh rác (chai nhựa, giấy, kim loại, thủy tinh, cardboard, trash...) "
            "để mô hình phân loại."
        ),
    )

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()

