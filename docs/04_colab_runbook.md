# Hướng dẫn chạy dự án trên Google Colab

## 1. Chuẩn bị trước khi lên Colab

- **Mã nguồn**: đồng bộ thư mục dự án `TrashProject/` lên GitHub hoặc nén `.zip` để tải lên Google Drive.
- **Dữ liệu**: chuẩn bị cấu trúc `data/raw/<class_name>/*.jpg`. Nếu dùng bộ TrashNet/TACO, tải trước để kiểm tra chất lượng ảnh.
- **Thông tin nhãn**: đảm bảo tên thư mục khớp với nhãn sử dụng trong báo cáo và code.
- **GPU**: vào `Runtime → Change runtime type` và chọn GPU (L4/T4/A100 tùy tài khoản).

## 2. Các bước chạy notebook `notebooks/trash_classifier_colab.ipynb`

1. Mount Google Drive nếu dữ liệu hoặc mô hình lưu trên Drive.
2. Cài đặt thư viện bằng cell `pip install -r requirements.txt` (Torch, Albumentations, Gradio, scikit-learn...).
3. Clone dự án từ GitHub (hoặc unzip từ Drive) vào `/content/TrashProject`.
4. (Tùy chọn) Tải TrashNet trực tiếp trong notebook:
   ```bash
   %cd /content/TrashProject
   !python scripts/download_trashnet.py --output-dir data/raw
   ```
   Thư mục sau khi tải xong có dạng `data/raw/<class_name>/*.jpg`.
5. Nếu sử dụng dataset khác (TACO, dữ liệu riêng), copy vào `data/raw` với cấu trúc thư mục theo lớp.
6. Chạy cell chia dữ liệu thành `train/val/test`.
7. Cấu hình huấn luyện (batch size, epochs, model_name, loss, scheduler...).
8. Gọi `trainer.train()` để bắt đầu huấn luyện và theo dõi log trực tiếp trên Colab.
9. Sau khi hoàn tất, hiển thị báo cáo và confusion matrix để đánh giá mô hình.
10. Chạy cell Gradio để tạo demo upload ảnh. Đặt `share=True` nếu muốn sinh link truy cập tạm thời.

## 3. Lưu và tải trọng số mô hình

- Trọng số tốt nhất nằm ở `artifacts/best.pt`. Sao chép về Drive để lưu trữ lâu dài.
- Các file bổ trợ:
  - `artifacts/history.pth`: log loss/accuracy theo epoch.
  - `artifacts/classification_report.pth`: dict chứa Precision/Recall/F1.
  - `artifacts/confusion_matrix.pth`: ma trận nhầm lẫn.

## 4. Checklist đánh giá

- [ ] Accuracy trên tập test đạt hoặc vượt mục tiêu (ví dụ 85%).
- [ ] Macro F1 từng lớp được báo cáo rõ ràng.
- [ ] Confusion matrix phân tách được các lớp bị nhầm lẫn nhiều.
- [ ] Nếu dữ liệu mất cân bằng, so sánh CrossEntropy và FocalLoss.
- [ ] So sánh ít nhất 2 kiến trúc (ResNet18, EfficientNet-B0, ViT) kèm thời gian huấn luyện và độ chính xác.

## 5. Gợi ý mở rộng dự án

- Thu thập thêm ảnh thực tế để cải thiện độ tổng quát.
- Kết hợp Mixup/CutMix cho tập train nếu có nhiều chủng loại vật thể.
- Thử Grid Search hoặc Optuna cho learning rate và batch size.
- Dùng Grad-CAM để minh họa vùng mô hình chú ý trong báo cáo.
- Nếu cần nhận dạng nhiều vật thể trong cùng ảnh, cân nhắc pipeline detection + classification.
