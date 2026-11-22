# Model Training & Evaluation Scripts

Thư mục này chỉ lưu **code** tiền xử lý, huấn luyện và kiểm thử (không lưu file trọng số `.pth`/`.ckpt`).

## Cấu trúc

- `tall/` — **TALL-Swin** (`rainy-xu/TALL4Deepfake`, model đang dùng cho web demo):
  - `colab_preprocess.py`: Tiền xử lý video & cắt khuôn mặt (chạy trên **Google Colab**).
  - `kaggle_train.py`: Huấn luyện `TALL_SWIN` (chạy trên **Kaggle GPU**).
  - `kaggle_test.py`: Đánh giá checkpoint `model_best.pth` trên tập test (chạy trên **Kaggle GPU**).
- `dfd/` — **DFD-FCG** (`aiiu-lab/DFD-FCG`, thử nghiệm đối chứng):
  - `colab_preprocess.py`: Tiền xử lý video (chạy trên **Google Colab**).
  - `kaggle_train.py`: Huấn luyện `DFD-FCG` (chạy trên **Kaggle T4 x2**).
  - `kaggle_test.py`: Đánh giá checkpoint `.ckpt` trên tập test (chạy trên **Kaggle GPU**).

## Lưu ý khi chạy Web Demo

File trọng số (`model_best.pth`) lưu trên Kaggle Dataset/Output hoặc Google Drive, sau đó trỏ biến môi trường `TALL_CKPT` khi chạy [backend/app.py](../backend/app.py).
