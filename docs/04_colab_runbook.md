# Hướng dẫn chạy dự án trên Google Colab

## 1. Chuẩn bị trước khi lên Colab

- **Mã nguồn**: đồng bộ thư mục dự án `TrashProject/` lên GitHub hoặc nén `.zip` để upload vào Drive.
- **Dữ liệu**: tạo thư mục `data/raw/<class_name>/*.jpg`. Nếu dùng TrashNet/TACO, tải về trước và kiểm tra chất lượng ảnh.
- **Thông tin nhãn**: đảm bảo tên thư mục khớp với nhãn bạn dùng trong báo cáo.
- **GPU**: chọn Runtime → Change runtime type → GPU (L4/T4/A100 tùy tài khoản).

## 2. Các bước chạy notebook `notebooks/trash_classifier_colab.ipynb`

1. Mount Google Drive nếu dữ liệu/model lưu trên đó.
2. Cài đặt phụ thuộc bằng cell pip (Torch, Albumentations, Gradio, scikit-learn...).
3. Clone dự án từ GitHub (hoặc unzip từ Drive) vào `/content/TrashProject`.
4. Tùy chọn tải dataset công khai (TrashNet, TACO) hoặc sử dụng dữ liệu của bạn trong `data/raw`.
5. Chạy cell chia dữ liệu thành `train/val/test`.
6. Cấu hình huấn luyện (batch size, epochs, model_name, lựa chọn loss).
7. Gọi `trainer.train()` để bắt đầu. Theo dõi log trực tiếp trong Colab.
8. Sau khi hoàn tất, hiển thị báo cáo và confusion matrix để đánh giá mô hình.
9. Chạy cell Gradio để tạo demo upload ảnh. Dùng `share=True` nếu muốn tạo link tạm thời.

## 3. Lưu và tải trọng số mô hình

- Trọng số tốt nhất nằm ở `artifacts/best.pt`. Sao chép về Drive để lưu trữ lâu dài.
- Các file bổ trợ:
  - `artifacts/history.pth`: log loss/accuracy theo epoch.
  - `artifacts/classification_report.pth`: dict chứa Precision/Recall/F1.
  - `artifacts/confusion_matrix.pth`: ma trận nhầm lẫn.

## 4. Checklist đánh giá

- [ ] Accuracy trên tập test ≥ ngưỡng mục tiêu (ví dụ 85%).
- [ ] Macro F1 từng lớp được báo cáo rõ ràng.
- [ ] Confusion matrix phân tích lớp bị nhầm lẫn.
- [ ] Nếu dữ liệu mất cân bằng, báo cáo có so sánh CrossEntropy vs FocalLoss.
- [ ] So sánh ít nhất 2 kiến trúc (ResNet18, EfficientNet-B0, ViT) về thời gian huấn luyện và độ chính xác.

## 5. Gợi ý mở rộng đồ án

- Thu thập thêm ảnh thực tế để cải thiện độ tổng quát.
- Kết hợp Mixup/CutMix cho tập train nếu có nhiều chồng chéo vật thể.
- Thử Grid Search hoặc Optuna cho learning rate và batch size.
- Dùng Grad-CAM để minh họa vùng mô hình chú ý trong báo cáo.
- Nếu cần nhận dạng nhiều vật thể trong cùng ảnh, chuyển sang pipeline detection + classification.

