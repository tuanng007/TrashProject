# TrashProject

Ứng dụng mẫu phân loại rác thải bằng học sâu, bao gồm:

- Tài liệu hướng dẫn lý thuyết và workflow (`docs/`).
- Notebook Colab để huấn luyện và demo (`notebooks/trash_classifier_colab.ipynb`).
- Mã nguồn huấn luyện PyTorch (`src/`) và script CLI (`scripts/train.py`).

## Chuẩn bị

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Huấn luyện cục bộ

```bash
python -m scripts.train --train-dir data/train --val-dir data/val --test-dir data/test --device cpu
```

Chi tiết xem thêm `docs/05_local_run.md`.

## Huấn luyện trên Colab

Mở `notebooks/trash_classifier_colab.ipynb` và chạy tuần tự các cell (mount Drive, cài gói, chuẩn bị dữ liệu, huấn luyện, demo Gradio). Runbook đầy đủ ở `docs/04_colab_runbook.md`.

