"""Gradio demo app for the waste classification models.

Chuc nang:
- Load checkpoint tot nhat (mac dinh: artifacts/best.pt).
- Su dung lai backbone (ResNet18/MobileNetV3/EfficientNetB0/ViT) va preprocessing
  giong pipeline huan luyen.
- Cho phep nguoi dung upload anh va xem top-k lop duoc du doan.
- Ho tro chon nhieu mo hinh trong UI de so sanh (artifacts/<model_name>/best.pt).
- Neu checkpoint co kem 'class_to_idx' va 'model_name' thi khong can --train-dir.

Vi du chay:
    # Single model (giong code cu)
    python -m scripts.demo_gradio \\
        --train-dir data/train \\
        --checkpoint artifacts/best.pt \\
        --model resnet18

    # Single model (checkpoint moi, khong can train-dir)
    python -m scripts.demo_gradio \\
        --checkpoint artifacts/best.pt \\
        --model auto

    # Multi-model so sanh
    python -m scripts.demo_gradio \\
        --train-dir data/train \\
        --models "resnet18,mobilenetv3,efficientnetb0,vitb16" \\
        --artifacts-root artifacts
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr
from PIL import Image
import torch

from src.training.dataset import WasteDataset, default_transforms
from src.training.trainer import build_model


CLASS_ICONS: Dict[str, str] = {
    # Mapping ten thu muc/lop -> icon de hien thi tren UI
    "cardboard": "📦",
    "glass": "🍾",
    "metal": "🥫",
    "paper": "📄",
    "plastic": "🧴",
    "trash": "🗑️",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio demo for waste classifier.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help=(
            "Thu muc train (de suy ra danh sach lop). "
            "Co the bo qua neu checkpoint co san 'class_to_idx'."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/best.pt"),
        help="Duong dan toi checkpoint da huan luyen (single model mode).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help=(
            "Ten mo hinh (resnet18/mobilenetv3/efficientnetb0/vitb16). "
            "Dung 'auto' de lay tu checkpoint neu co."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Danh sach cac mo hinh (phan tach boi dau phay, "
            "vd: 'resnet18,mobilenetv3,efficientnetb0,vitb16') de so sanh trong Gradio. "
            "Neu khong truyen, script chay o che do single model."
        ),
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
        help=(
            "Thu muc goc chua artifact (mac dinh artifacts/<model_name>/best.pt). "
            "Dung cho multi-model mode."
        ),
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Kich thuoc anh dau vao (phai khop voi luc huan luyen).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiet bi suy luan (cuda/cpu).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Bat share=True de Gradio tao link public tam thoi.",
    )
    return parser.parse_args()


def format_scores(probs: torch.Tensor, idx_to_class: Dict[int, str]) -> Dict[str, float]:
    """Chuyen vector xac suat thanh dict label (co icon) -> score."""
    scores: Dict[str, float] = {}
    for idx in range(len(idx_to_class)):
        class_name = idx_to_class[idx]
        icon = CLASS_ICONS.get(class_name, "")
        label = f"{icon} {class_name}" if icon else class_name
        scores[label] = float(probs[idx])
    return scores


def load_artifacts(
    train_dir: Optional[Path],
    checkpoint: Path,
    model_name: str,
    img_size: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, callable, Dict[int, str]]:
    state = torch.load(checkpoint, map_location=device)

    class_to_idx = None
    if isinstance(state, dict) and "class_to_idx" in state:
        class_to_idx = state["class_to_idx"]
    if not class_to_idx:
        if train_dir is None:
            raise ValueError(
                "Can --train-dir de suy ra danh sach lop, "
                "hoac dung checkpoint moi co san 'class_to_idx'."
            )
        dataset = WasteDataset(train_dir)
        class_to_idx = dataset.class_to_idx

    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    num_classes = len(idx_to_class)

    if model_name.strip().lower() == "auto":
        if not (isinstance(state, dict) and state.get("model_name")):
            raise ValueError("Checkpoint khong co 'model_name'; hay truyen --model.")
        model_name = str(state["model_name"])

    # Build model va load weights
    model = build_model(model_name, num_classes)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        # Truong hop checkpoint chi chua state_dict thuan
        model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Dang eval transform giong pipeline huan luyen
    _, eval_tf, _ = default_transforms(img_size)

    return model, eval_tf, idx_to_class


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Multi-model mode: --models duoc cung cap
    models_arg = (args.models or "").strip()
    if models_arg:
        model_names = [m.strip() for m in models_arg.split(",") if m.strip()]
    else:
        model_names = []

    if model_names:
        loaded_models: Dict[str, torch.nn.Module] = {}
        eval_tf = None
        idx_to_class = None

        for model_name in model_names:
            ckpt_path = args.artifacts_root / model_name / "best.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Khong tim thay checkpoint cho mo hinh '{model_name}' "
                    f"tai {ckpt_path}. Hay chay huan luyen va chinh lai --artifacts-root hoac ten model."
                )
            model, eval_tf_local, idx_to_class_local = load_artifacts(
                train_dir=args.train_dir,
                checkpoint=ckpt_path,
                model_name=model_name,
                img_size=args.img_size,
                device=device,
            )
            loaded_models[model_name] = model
            # Dung chung eval_tf va idx_to_class (gia su cung dataset / mapping lop)
            if eval_tf is None:
                eval_tf = eval_tf_local
            if idx_to_class is None:
                idx_to_class = idx_to_class_local

        assert eval_tf is not None and idx_to_class is not None

        def predict(image: Image.Image, model_name: str):
            if image is None:
                return {}
            if model_name not in loaded_models:
                return {}
            x = eval_tf(image).unsqueeze(0).to(device)
            model = loaded_models[model_name]
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu()

            return format_scores(probs, idx_to_class)

        demo = gr.Interface(
            fn=predict,
            inputs=[
                gr.Image(type="pil", label="Upload anh rac"),
                gr.Dropdown(
                    choices=model_names,
                    value=model_names[0],
                    label="Chon mo hinh",
                ),
            ],
            outputs=gr.Label(num_top_classes=6, label="Du doan loai rac (top 6)"),
            title="Phan loai rac - So sanh nhieu mo hinh",
            description=(
                "Demo phan loai rac tren nhieu mo hinh khac nhau.\n\n"
                "1. Upload mot anh rac (chai nhua, giay, kim loai, thuy tinh, cardboard, trash...).\n"
                "2. Chon mo hinh o dropdown ben duoi (ResNet18, MobileNetV3, EfficientNetB0, ViT...).\n"
                "3. Xem top-6 lop duoc du doan kem xac suat de so sanh."
            ),
        )
    else:
        # Single model mode (giu nguyen hanh vi cu)
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

            # Tra ve dict {label (co icon): probability}
            return format_scores(probs, idx_to_class)

        demo = gr.Interface(
            fn=predict,
            inputs=gr.Image(type="pil", label="Upload anh rac"),
            outputs=gr.Label(num_top_classes=6, label="Du doan loai rac (top 6)"),
            title="Phan loai rac bang hoc sau",
            description=(
                "Demo phan loai anh rac bang mo hinh hoc sau da fine-tune.\n\n"
                "Upload mot anh rac (chai nhua, giay, kim loai, thuy tinh, cardboard, trash...) "
                "de xem top-6 lop mo hinh du doan kem xac suat."
            ),
        )

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
