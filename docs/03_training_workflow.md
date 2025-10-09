# Quy trình xây dựng hệ thống phân loại rác thải bằng AI

Tài liệu này giúp bạn định hướng toàn bộ pipeline từ nghiên cứu bài toán cho đến triển khai demo web/app có thể chạy trên Google Colab hoặc môi trường tương đương.

## 1. Kiến thức nền cần nắm

- **Phân loại rác**: phân nhóm theo tính chất (hữu cơ, nhựa, giấy, kim loại, thủy tinh, vải, pin...). Quy định địa phương có thể khác nhau, vì vậy cần thống nhất bộ nhãn sử dụng trong dữ liệu.
- **Bài toán học máy**:
  - *Multi-class*: mỗi ảnh thuộc đúng một lớp nhãn. Ví dụ ảnh chai nhựa → lớp "Plastic".
  - *Multi-label*: mỗi ảnh có thể thuộc nhiều lớp (ít gặp với rác thải nhưng hữu ích nếu ảnh chứa nhiều vật thể). Luồng xử lý tương tự nhưng dùng hàm kích hoạt sigmoid và loss dạng binary.
- **Xử lý dữ liệu ảnh**:
  - *Resize & chuẩn hóa* để đưa ảnh về kích thước cố định (ví dụ `224×224`) và chuẩn giá trị pixel theo thang `[0,1]` hoặc phân phối chuẩn (`mean`, `std` của ImageNet).
  - *Cân bằng lớp*: dùng `class weighting`, `focal loss`, oversampling hoặc augmentation có chủ đích để giảm bias giữa lớp phổ biến và lớp hiếm.
- **Data augmentation theo ngữ cảnh**: xoay (rotate), lật (flip), crop ngẫu nhiên, blur, jitter màu, `Cutout/Random Erasing`, `Mixup/CutMix`. Cần đảm bảo phép biến đổi phù hợp thực tế (không làm mất dấu hiệu nhận dạng).
- **Các kiến trúc phân loại ảnh**:
  - *CNN*: ResNet, EfficientNet, MobileNet (nhẹ, phù hợp triển khai trên thiết bị di động).
  - *Transformer*: Vision Transformer (ViT), SwinTransformer — độ chính xác cao nhưng cần nhiều dữ liệu/tài nguyên.
  - So sánh tốc độ và độ chính xác để chọn mô hình tối ưu cho yêu cầu triển khai.
- **Phát hiện đối tượng**: dùng YOLOv8, Faster R-CNN, DETR nếu cần tách vật thể trước khi phân loại từng loại rác.
- **Few-shot / Transfer learning**: fine-tune layer cuối, freeze backbone, lựa chọn learning rate schedule (Cosine, OneCycle, StepLR).
- **Chỉ số đánh giá**: Accuracy, Precision/Recall/F1 (macro để cân lớp), Confusion Matrix, ROC-AUC (nếu multi-label), mAP (cho detection).

## 2. Chuẩn bị dữ liệu

1. **Thu thập bộ dữ liệu**:
   - Bộ công khai: [TrashNet](https://github.com/garythung/trashnet), [TACO](https://github.com/pedropro/TACO). Có thể bổ sung ảnh chụp thực tế để tăng tính đa dạng.
   - Chuẩn hóa cấu trúc thư mục: `dataset/train/<class_name>/*.jpg`, `dataset/val/<class_name>/*.jpg`.
2. **Tiền xử lý**:
   - Tạo script kiểm tra và loại bỏ ảnh lỗi.
   - Ghi lại `class_counts.json` để phục vụ tính trọng số lớp.
   - Tạo file `labels_map.json` mô tả nhãn → id.
3. **Augmentation**:
   - Sử dụng `torchvision.transforms`, Albumentations hoặc Keras preprocessing.
   - Tách pipeline cho train/val/test (train có augmentation mạnh hơn).
4. **Lưu trữ trên Colab**:
   - Upload lên Google Drive, tạo symlink trong Colab (`/content/drive/MyDrive/...`).
   - Hoặc dùng `kaggle datasets download` nếu bộ dữ liệu đã được đăng lên Kaggle.

## 3. Xây dựng mô hình & huấn luyện

### 3.1 Cấu trúc thư mục mã nguồn

```
TrashProject/
├── docs/
│   └── 03_training_workflow.md
├── notebooks/
│   └── trash_classifier_colab.ipynb   # Notebook chính chạy trên Colab
├── src/
│   ├── training/
│   │   ├── dataset.py                 # Dataset + transforms
│   │   ├── losses.py                  # Loss functions (CrossEntropy, Focal)
│   │   ├── optim.py                   # Optimizer & scheduler factory
│   │   └── trainer.py                 # Pipeline huấn luyện (PyTorch Lightning/thuần)
│   └── utils/
│       └── metrics.py                 # Accuracy, Precision/Recall/F1, Confusion Matrix
└── requirements.txt
```

### 3.2 Lựa chọn mô hình

- **Baseline**: dùng `torchvision.models` (ResNet18) với head được tùy biến lại cho số lớp tương ứng.
- **Model nhẹ**: MobileNetV3, EfficientNet-B0 (dễ deploy lên web/mobile).
- **Model mạnh**: ViT-B/16 (cần GPU mạnh trên Colab Pro hoặc T4/A100).
- Có thể chạy nhiều mô hình trong Colab và lưu kết quả vào CSV để so sánh.

### 3.3 Huấn luyện

1. Chia train/val/test (ví dụ 70/20/10). Duy trì stratified split.
2. Chọn `batch_size` dựa trên VRAM (32-128).
3. Dùng optimizer SGD hoặc AdamW. Learning rate ban đầu 3e-4 (AdamW) hoặc 0.01 (SGD + momentum).
4. Scheduler: CosineAnnealingLR, OneCycleLR hoặc StepLR.
5. Loss: CrossEntropy + class weights, hoặc FocalLoss nếu dữ liệu mất cân bằng mạnh.
6. Log metrics bằng `TensorBoard` hoặc `Weights & Biases`.

### 3.4 Fine-tuning

- Bắt đầu với mô hình pretrained, freeze backbone (feature extractor) trong 5-10 epoch đầu.
- Unfreeze toàn bộ và giảm learning rate cho backbone (ví dụ `lr_backbone = 1e-4`, `lr_head = 1e-3`).
- Sử dụng Gradual Unfreezing hoặc Layer-wise LR decay cho mô hình transformer.

## 4. Đánh giá và so sánh mô hình

1. Chạy inference trên tập test, tính Accuracy, Macro Precision/Recall/F1, ghi vào `results.json`.
2. Vẽ Confusion Matrix (sklearn) để xem lớp nào bị nhầm lẫn.
3. Với detection, dùng mAP@0.5 và kiểm tra bounding box trên một số ảnh.
4. So sánh kết quả giữa các mô hình, cân nhắc trade-off tốc độ/độ chính xác/phức tạp triển khai.
5. Ghi chú hạn chế và đề xuất cải thiện (thu thập thêm dữ liệu, augmentation mạnh hơn, mixup/cutmix, pseudo-labeling...).

## 5. Triển khai demo

- Chọn framework nhẹ trên Colab: Streamlit, Gradio, hoặc Flask (chạy bằng `ngrok`).
- Workflow đề xuất với Gradio:
  1. Tải mô hình `.pt` từ Drive.
  2. Tạo app `gradio.Interface(fn=predict, inputs=Image, outputs=Label/JSON)`.
  3. Cho phép người dùng upload ảnh, hiển thị nhãn & độ tự tin.
  4. Có thể hiển thị Grad-CAM để minh họa vùng mô hình quan tâm.
- Đóng gói scripts `inference.py` với pipeline preprocess → model → postprocess.

## 6. Lộ trình làm đồ án

| Tuần | Nội dung | Ghi chú |
|------|----------|---------|
| 1 | Nghiên cứu tài liệu, thống nhất hệ nhãn, thu thập dữ liệu | Tạo checklist chất lượng dữ liệu |
| 2 | Hoàn thiện tiền xử lý, augmentation, baseline training | Lưu log/trọng số mô hình |
| 3 | Thử nghiệm mô hình nâng cao, tinh chỉnh siêu tham số | So sánh các kiến trúc khác nhau |
| 4 | Đánh giá chi tiết, xây dựng demo web/app | Chuẩn bị slide/báo cáo cuối kỳ |

## 7. Checklist chuẩn bị cho Google Colab

- [ ] Google Drive chứa `dataset.zip`, `models/`.
- [ ] Notebook Colab với các bước: mount Drive, cài phụ thuộc, chuẩn bị dữ liệu, train, evaluate, export model, demo.
- [ ] File `requirements.txt` (PyTorch, torchvision, albumentations, timm, gradio...).
- [ ] Script tải trọng số mô hình và chạy thử 5 ảnh mẫu.
- [ ] Báo cáo tổng hợp kết quả (Accuracy, bảng F1 từng lớp, Confusion Matrix).

> **Lưu ý**: Nếu lớp cực kỳ mất cân bằng, cân nhắc bài toán multi-label với thresholding hoặc tạo pipeline detection + classification để tách nhiều vật thể trong cùng khung hình.

