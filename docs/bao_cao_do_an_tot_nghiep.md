# BÁO CÁO ĐỒ ÁN TỐT NGHIỆP

Đề tài: Xây dựng hệ thống phân loại rác thải bằng Trí tuệ nhân tạo (AI)

Sinh viên thực hiện: [Điền tên]

Giáo viên hướng dẫn: [Điền tên]

Thời gian thực hiện: [Điền thời gian]


## Chương 1: Tổng quan và cơ sở lý thuyết

1.1. Bối cảnh và ý nghĩa

- Quản lý rác thải là một bài toán quan trọng của đô thị thông minh và phát triển bền vững. Tự động hóa khâu phân loại rác giúp giảm chi phí nhân công, nâng cao tỷ lệ tái chế và giảm ô nhiễm môi trường.
- Hệ thống phân loại rác dựa trên AI sử dụng học sâu để nhận diện vật thể trong ảnh (chai nhựa, giấy, kim loại, thủy tinh, hữu cơ, vải, pin, …), từ đó đưa ra nhãn phân loại phù hợp.

1.2. Mục tiêu đề tài

- Xây dựng pipeline hoàn chỉnh cho bài toán phân loại ảnh rác thải: chuẩn bị dữ liệu, huấn luyện, đánh giá, và demo suy luận.
- Thiết kế mã nguồn có cấu trúc, dễ mở rộng, hỗ trợ nhiều kiến trúc mô hình khác nhau (ResNet18, MobileNetV3, EfficientNet-B0) và có thể triển khai trên Google Colab và máy cục bộ.
- Đạt mức độ chính xác và F1 macro đạt/tiệm cận mục tiêu (ví dụ ≥ 85% tùy tập dữ liệu), có báo cáo đánh giá theo từng lớp.

1.3. Phạm vi và giới hạn

- Phạm vi: Phân loại ảnh tĩnh một nhãn (multi-class). Có thể mở rộng multi-label hoặc kết hợp nhận dạng đa vật thể trong tương lai.
- Giới hạn: Chất lượng dữ liệu công khai (TrashNet, TACO) chưa đồng nhất, có thể gây mất cân bằng lớp; tài nguyên tính toán phụ thuộc GPU Colab hoặc GPU cục bộ; tốc độ suy luận phụ thuộc kiến trúc mô hình.

1.4. Tổng quan cơ sở lý thuyết

- Bài toán phân loại ảnh: ánh xạ ảnh đầu vào sang một nhãn lớp; thường dùng mạng CNN hoặc Transformer thị giác.
- Chia tập dữ liệu: train/val/test; cần giữ phân phối tương đồng (stratified split) để đánh giá khách quan.
- Tiền xử lý và tăng cường dữ liệu (augmentation): Resize, crop, flip, rotate, jitter màu, Cutout/Mixup/CutMix để tăng đa dạng và chống overfitting.
- Học chuyển giao (transfer learning) và fine-tuning: sử dụng trọng số pretrained (ImageNet), đóng băng backbone ở giai đoạn đầu, sau đó unfreeze có kiểm soát.
- Thước đo đánh giá: Accuracy, Precision/Recall/F1 (macro để cân bằng lớp), Confusion Matrix; nếu mở rộng detection có thể dùng mAP.

1.5. Cấu trúc báo cáo

- Chương 1: Trình bày bối cảnh, mục tiêu, phạm vi và các khái niệm nền tảng.
- Chương 2: Cơ sở lý thuyết chi tiết và công nghệ sử dụng trong đồ án.
- Chương 3: Phân tích yêu cầu, kiến trúc và thiết kế hệ thống.
- Chương 4: Triển khai trên máy cục bộ/Colab, hướng dẫn chạy và demo.
- Chương 5: Đánh giá kết quả, hạn chế và hướng phát triển.


## Chương 2: Cơ sở lý thuyết và công nghệ sử dụng

2.1. Phân loại ảnh và các hướng tiếp cận

- CNN (Convolutional Neural Networks): ResNet, EfficientNet, MobileNet; ưu điểm tốc độ tốt, phù hợp triển khai trên thiết bị hạn chế tài nguyên.
- Transformer cho thị giác: ViT, SwinTransformer; độ chính xác cao nhưng yêu cầu dữ liệu/tài nguyên lớn.
- So sánh và lựa chọn mô hình theo tiêu chí: độ chính xác, tốc độ suy luận, kích thước mô hình, tài nguyên huấn luyện.

2.2. Tiền xử lý dữ liệu và augmentation

- Chuẩn hóa kích thước và pixel theo phân phối ImageNet; tăng cường theo ngữ cảnh (xoay, lật, crop, jitter màu, blur) và các kỹ thuật Mixup/CutMix khi phù hợp.
- Cân bằng lớp: class weighting, focal loss, oversampling; lưu ý giữ tính thực tế của biến đổi.

2.3. Học chuyển giao, tối ưu và lịch học

- Fine-tuning từng phần: freeze backbone vài epoch đầu, sau đó unfreeze dần; điều chỉnh LR theo layer hoặc backbone/head khác nhau.
- Tối ưu hóa: AdamW/SGD; lịch học OneCycle, Cosine Annealing, Step.

2.4. Chỉ số đánh giá

- Accuracy, Precision/Recall/F1 (macro) để phản ánh hiệu năng trong bối cảnh mất cân bằng lớp.
- Confusion Matrix để phân tích nhầm lẫn giữa các lớp cụ thể.

2.5. Công nghệ sử dụng trong dự án

- Ngôn ngữ và môi trường: Python 3.10+, virtualenv.
- Thư viện chính: PyTorch, Torchvision, Albumentations, timm, scikit-learn, Grad-CAM, Gradio.
- Yêu cầu hệ thống: GPU NVIDIA khuyến nghị cho huấn luyện; Colab hỗ trợ nhanh chóng cho demo/training.
- Tham chiếu mã nguồn và tài liệu:
  - Cấu trúc workflow và checklist: `docs/03_training_workflow.md:1`
  - Runbook Colab: `docs/04_colab_runbook.md:1`
  - Hướng dẫn chạy cục bộ: `docs/05_local_run.md:1`
  - Danh sách phụ thuộc: `requirements.txt:1`


## Chương 3: Phân tích và thiết kế hệ thống

3.1. Yêu cầu chức năng

- Chuẩn bị và nạp dữ liệu theo cấu trúc thư mục `train/val/test`.
- Huấn luyện mô hình với lựa chọn kiến trúc, loss, optimizer, scheduler; theo dõi và lưu lịch sử.
- Đánh giá trên tập validation/test; xuất báo cáo phân loại và ma trận nhầm lẫn.
- Suy luận và demo thông qua Gradio/Streamlit (tùy chọn).

3.2. Yêu cầu phi chức năng

- Độ chính xác và F1 macro đạt mục tiêu đề ra trên tập test.
- Thời gian suy luận hợp lý (phụ thuộc mô hình), kích thước mô hình vừa phải.
- Mã nguồn rõ ràng, tách mô-đun; dễ mở rộng thêm kiến trúc/loss.

3.3. Kiến trúc tổng thể hệ thống

- Thành phần Dữ liệu: Dataset, DataLoader, transforms. Tham chiếu: `src/training/dataset.py:1`
- Thành phần Mô hình: factory xây dựng mô hình từ Torchvision. Tham chiếu: `src/training/trainer.py:1`
- Thành phần Tối ưu: optimizer/scheduler builder. Tham chiếu: `src/training/optim.py:1`
- Thành phần Hàm mất mát: CrossEntropy, Focal + class weighting. Tham chiếu: `src/training/losses.py:1`
- Thành phần Đánh giá: accuracy, classification report, confusion matrix. Tham chiếu: `src/utils/metrics.py:1`
- Giao diện dòng lệnh (CLI): huấn luyện cục bộ. Tham chiếu: `scripts/train.py:1`

3.4. Thiết kế dữ liệu và tiền xử lý

- Cấu trúc thư mục dữ liệu: `data/train/<class_name>`, `data/val/<class_name>`, `data/test/<class_name>`.
- Mapping lớp: sinh tự động từ thư mục train và dùng thống nhất cho val/test.
- Bộ biến đổi (transforms): khác nhau giữa train (mạnh hơn) và eval (chuẩn hóa trung lập).

3.5. Thiết kế mô-đun chính

- `dataset.py`: Khai báo `WasteDataset`, `DataConfig`, `default_transforms`, `create_dataloaders` cho phép tái sử dụng và cấu hình linh hoạt kích thước ảnh, batch size, số worker.
- `losses.py`: `LossConfig`, `compute_class_weights`, `focal_loss`, `build_loss` giúp xử lý mất cân bằng lớp.
- `optim.py`: `OptimConfig`, `SchedulerConfig`, `build_optimizer`, `build_scheduler` hỗ trợ OneCycle, Cosine, Step.
- `trainer.py`: `TrainConfig`, `build_model` (ResNet18/MobileNetV3/EfficientNetB0), lớp `WasteTrainer` đóng gói huấn luyện, đánh giá, lưu checkpoint và báo cáo.
- `metrics.py`: Tận dụng scikit-learn để xuất báo cáo phân loại và ma trận nhầm lẫn, đồng thời có hàm accuracy nhanh.

3.6. Luồng xử lý chính

- Chuẩn bị DataLoader → Khởi tạo mô hình (pretrained) → Cấu hình tối ưu/lịch học → Vòng lặp huấn luyện (forward, loss, backward, step) → Đánh giá định kỳ trên val → Lưu checkpoint tốt nhất → Test cuối cùng và lưu báo cáo.


## Chương 4: Triển khai hệ thống

4.1. Chuẩn bị môi trường

- Tạo môi trường ảo và cài phụ thuộc:

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Tham chiếu hướng dẫn: `README.md:1`, `docs/05_local_run.md:1`, `requirements.txt:1`.

4.2. Chuẩn bị dữ liệu

- Tạo cấu trúc thư mục `data/train/<class>`, `data/val/<class>`, tùy chọn `data/test/<class>`.
- Có thể sử dụng TrashNet/TACO hoặc dữ liệu tự thu thập; kiểm tra chất lượng và cân bằng lớp.

4.3. Huấn luyện trên máy cục bộ (CLI)

Ví dụ lệnh:

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

Kết quả lưu tại `artifacts/`: `best.pt`, `history.pth`, `classification_report.pth`, `confusion_matrix.pth`.

4.4. Huấn luyện và demo trên Google Colab

- Mở notebook `notebooks/trash_classifier_colab.ipynb` và làm theo runbook: `docs/04_colab_runbook.md:1`.
- Các bước chính: mount Drive → cài gói → chuẩn bị dữ liệu → khai báo danh sách mô hình → huấn luyện so sánh → xuất báo cáo → demo Gradio (tùy chọn `share=True`).

4.5. Demo suy luận nhanh (mẫu code)

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

4.6. Quản lý mô hình và artifact

- Checkpoint tốt nhất: `artifacts/best.pt`.
- Lịch sử huấn luyện: `artifacts/history.pth`.
- Báo cáo phân loại và ma trận nhầm lẫn: `artifacts/classification_report.pth`, `artifacts/confusion_matrix.pth`.


## Chương 5: Đánh giá và đề xuất phát triển

5.1. Phương pháp đánh giá

- Đo trên tập test độc lập (nếu có), báo cáo Accuracy, Precision/Recall/F1 (macro), Confusion Matrix.
- So sánh giữa các kiến trúc (ResNet18, MobileNetV3, EfficientNet-B0) theo tiêu chí độ chính xác, thời gian huấn luyện và kích thước mô hình.

5.2. Kết quả thực nghiệm (điền số liệu)

- Bảng tổng hợp: Accuracy, Macro F1 của từng mô hình.
- Báo cáo chi tiết từng lớp: trích từ `classification_report.pth`.
- Ma trận nhầm lẫn: trích từ `confusion_matrix.pth` (có thể vẽ lại bằng matplotlib/seaborn).

5.3. Thảo luận

- Ưu điểm: Pipeline rõ ràng, mô-đun hóa; dễ thay mô hình/loss; hỗ trợ cả Colab và local; có artifacts phục vụ tái lập kết quả.
- Hạn chế: Phụ thuộc chất lượng và cân bằng dữ liệu; một số mô hình lớn yêu cầu GPU mạnh; kết quả có thể nhạy với tham số siêu.

5.4. Đề xuất phát triển

- Dữ liệu: Thu thập/chuẩn hóa thêm dữ liệu thực tế; cân bằng lớp tốt hơn; áp dụng augmentation theo ngữ cảnh mạnh hơn (Mixup/CutMix, Random Erasing).
- Mô hình: Thử ViT/Swin với LR decay theo layer; kỹ thuật unfreeze dần (Gradual Unfreezing); tối ưu hoá hyperparameters với Optuna.
- Tính năng: Mở rộng bài toán multi-label; kết hợp detection + classification để nhận nhiều vật thể trong cùng ảnh (YOLOv8/DETR + classifier).
- Triển khai: Đóng gói API (Flask/FastAPI), demo web nhẹ (Gradio/Streamlit), cân nhắc Mobile/Edge (TFLite/ONNX, pruning/quantization) để suy luận nhanh.
- Vận hành: Tổ chức MLOps cơ bản (theo dõi dataset/model version, logging metrics, reproducibility).


## Tài liệu tham khảo

- TrashNet Dataset: https://github.com/garythung/trashnet
- TACO Dataset: https://github.com/pedropro/TACO
- PyTorch: https://pytorch.org
- Torchvision Models: https://pytorch.org/vision/stable/models.html
- Albumentations: https://albumentations.ai
- Gradio: https://gradio.app


## Phụ lục

- Cấu trúc mã nguồn và tài liệu:
  - `src/training/dataset.py:1`, `src/training/losses.py:1`, `src/training/optim.py:1`, `src/training/trainer.py:1`
  - `src/utils/metrics.py:1`, `scripts/train.py:1`
  - `docs/03_training_workflow.md:1`, `docs/04_colab_runbook.md:1`, `docs/05_local_run.md:1`

