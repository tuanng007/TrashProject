# Hướng dẫn chạy dự án trên máy cục bộ

## 1. Chuẩn bị môi trường

- Cài đặt Python 3.10+.
- (Khuyến nghị) tạo virtualenv:

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS
```

- Cài phụ thuộc:

```bash
pip install -r requirements.txt
```

Nếu GPU NVIDIA, cài đúng phiên bản CUDA tương ứng với PyTorch (xem https://pytorch.org/get-started/locally/).

## 2. Chuẩn bị dữ liệu

- Tạo thư mục `data/train/<class_name>`, `data/val/<class_name>`, tùy chọn `data/test/<class_name>`.
- Có thể dùng script chia dữ liệu trong notebook hoặc tự chia bằng tay.

## 3. Chạy huấn luyện từ dòng lệnh

Script CLI nằm ở `scripts/train.py`. Ví dụ:

```bash
python scripts/train.py ^
  --train-dir data/train ^
  --val-dir data/val ^
  --test-dir data/test ^
  --epochs 20 ^
  --model efficientnetb0 ^
  --loss focal ^
  --scheduler cosine
```

Thông số quan trọng:
- `--model`: `resnet18`, `mobilenetv3`, `efficientnetb0`.
- `--loss`: `cross_entropy` hoặc `focal`.
- `--scheduler`: `onecycle`, `cosine`, `step`, hoặc `none`.
- `--device`: mặc định tự chọn `cuda` nếu có GPU.

Kết quả huấn luyện được lưu tại `artifacts/`:
- `best.pt`: checkpoint tốt nhất.
- `history.pth`: log loss/accuracy.
- `classification_report.pth`, `confusion_matrix.pth`: số liệu đánh giá.

## 4. Chạy inference nhanh

```python
from PIL import Image
import torch
from torchvision import transforms
from src.training.trainer import build_model

checkpoint = torch.load("artifacts/best.pt", map_location="cpu")
model = build_model("resnet18", num_classes=6)
model.load_state_dict(checkpoint["model_state"])
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = preprocess(Image.open("sample.jpg")).unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(image), dim=1)
print(probs)
```

## 5. Lưu ý

- Với GPU cục bộ, giảm `batch_size` nếu gặp lỗi thiếu VRAM.
- Nếu chạy CPU, tăng `epochs` và chấp nhận thời gian huấn luyện lâu hơn.
- Có thể kích hoạt TensorBoard bằng cách log thêm vào trainer hoặc dùng W&B tùy chọn.

