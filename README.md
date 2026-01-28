# YOLO Model Zoo

Đây là một project "Model Zoo" được xây dựng linh hoạt dựa trên thư viện **Ultralytics**, cho phép quản lý, huấn luyện và sử dụng nhiều phiên bản mô hình YOLO một cách có hệ thống.

## ✨ Tính năng nổi bật

- **Cấu hình động (Dynamic Configuration)**: Kế thừa và ghi đè các file cấu hình YAML để dễ dàng tạo và quản lý các thử nghiệm (experiment).
- **Quản lý dữ liệu với DVC**: Tự động tải xuống (pull) các bộ dữ liệu và trọng số mô hình cần thiết khi chạy, giúp đồng bộ môi trường giữa các thành viên.
- **Kiến trúc module hóa**: Dễ dàng mở rộng, đăng ký thêm các model hoặc wrapper mới thông qua `Registry`.
- **Giao diện dòng lệnh (CLI)**: Cung cấp các script tiện ích để thực hiện các tác vụ phổ biến: `train`, `infer`, `test`, `export` với các đối số được rút gọn.
- **Tích hợp sẵn các model**: Bao gồm các model `yolo11m`, `yolo26m` cho phát hiện vật thể và `yolo26m-seg` cho phân vùng ảnh.

---

## 🏗️ Cấu trúc thư mục

```
/
├─── configs/             # Chứa các file cấu hình YAML cho experiments
│    ├─── _base_/         # Các file cấu hình cơ sở (dataset, model, schedule)
│    └─── v11/, v26/      # Các file cấu hình cho từng phiên bản model cụ thể
├─── data/                # Nơi lưu trữ datasets (quản lý bởi DVC)
├─── models/              # Nơi lưu trữ trọng số model (.pt) (quản lý bởi DVC)
├─── runs/                # Thư mục output mặc định cho training và inference
├─── src/                 # Mã nguồn chính của framework
│    ├─── core/           # Các thành phần cốt lõi (config parser, data manager, registry)
│    └─── modeling/       # Nơi định nghĩa các model wrapper (vd: YOLO wrapper)
├─── tools/               # Các script để tương tác (train, infer, test, export)
├─── requirements.txt     # Các thư viện Python cần thiết
└─── README.md            # File hướng dẫn này
```

---

## 🚀 Bắt đầu nhanh

Phần này hướng dẫn cách sử dụng các script trong thư mục `tools`. Các lệnh đều hỗ trợ cả phiên bản đối số đầy đủ (ví dụ: `--config`) và viết tắt (ví dụ: `-c`).

### 1. Cài đặt môi trường

Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

Cấu hình DVC remote (chỉ cần lần đầu):
*Project này đã được cấu hình sẵn để sử dụng Google Drive. Bạn có thể cần xác thực tài khoản Google trong lần đầu tiên pull dữ liệu.*

### 2. Huấn luyện (Training)

Để bắt đầu một lần huấn luyện, hãy sử dụng script `tools/train.py` và chỉ định file cấu hình experiment.

**Ví dụ:** Huấn luyện model `yolo26m` với cấu hình demo.
```bash
python tools/train.py -c configs/v26/v26_m_demo.yaml
```
- **Tự động tải dữ liệu**: Script sẽ tự động kiểm tra và `dvc pull` bộ dữ liệu `coco_min` nếu nó chưa tồn tại.
- **Kết quả**: Kết quả sẽ được lưu vào thư mục `runs/detect/train/v26_m_demo_run/`.

### 3. Suy luận (Inference)

Sử dụng `tools/infer.py` để chạy dự đoán trên một ảnh hoặc video.

**Ví dụ:** Chạy inference với model `yolo11m` trên ảnh `dog_and_bike.jpeg`.
```bash
python tools/infer.py -c configs/v11/v11_m_demo.yaml -s data/raw/dog_and_bike.jpeg
```
- **Tự động tải trọng số**: Script sẽ tự động `dvc pull` file `yolo11m.pt` nếu nó chưa có sẵn.
- **Tùy chọn trọng số**: Bạn có thể chỉ định một file trọng số khác (ví dụ, kết quả từ quá trình training) bằng cờ `-w` (hoặc `--weights`):
  ```bash
  python tools/infer.py -c configs/v11/v11_m_demo.yaml -s data/raw/dog_and_bike.jpeg -w runs/detect/train/v26_m_demo_run/weights/best.pt
  ```
- **Kết quả**: Ảnh output sẽ được lưu trong `runs/detect/infer_result/`.

### 4. Đánh giá (Evaluation)

Sử dụng `tools/test.py` để đánh giá hiệu năng (mAP) của một model trên tập validation. Script sẽ tự động tải dataset và trọng số cần thiết (nếu được quản lý bởi DVC).

**Ví dụ:** Đánh giá model theo cấu hình `v11_m_demo.yaml`.
```bash
python tools/test.py -c configs/v11/v11_m_demo.yaml
```
- **Tùy chọn trọng số**: Bạn có thể đánh giá một file trọng số cụ thể (thay vì file được chỉ định trong config) bằng cờ `-w` (hoặc `--weights`):
  ```bash
  python tools/test.py -c configs/v11/v11_m_demo.yaml -w path/to/your/custom/weights.pt
  ```
- **Kết quả**: Các chỉ số mAP sẽ được in ra màn hình và lưu vào thư mục `runs/detect/eval_run_eval/`.

### 5. Xuất model (Export)

Sử dụng `tools/export.py` để chuyển đổi trọng số `.pt` sang các định dạng khác như ONNX hoặc TensorRT engine.

**Ví dụ:** Xuất model theo cấu hình `v26_m_demo.yaml` sang định dạng ONNX.
```bash
python tools/export.py -c configs/v26/v26_m_demo.yaml -f onnx
```
- **Tùy chọn trọng số**: Tương tự như các script khác, bạn có thể chỉ định một file trọng số cụ thể với cờ `-w` (hoặc `--weights`):
  ```bash
  python tools/export.py -c configs/v26/v26_m_demo.yaml -w path/to/your/weights.pt -f onnx
  ```
- **Kết quả**: File đã xuất (ví dụ `.onnx`) sẽ được tạo ra trong cùng thư mục với file trọng số đầu vào.

---

## ⚙️ Hệ thống Cấu hình

Hệ thống cho phép bạn tạo các file cấu hình experiment (ví dụ `v26_m_demo.yaml`) bằng cách kế thừa từ các file *base* và chỉ ghi đè những tham số cần thiết.

- **`base`**: Một danh sách các file cấu hình cơ sở cần gộp lại. Thứ tự rất quan trọng, file sau sẽ ghi đè lên file trước nếu có tham số trùng lặp.
- **Ghi đè (Override)**: Bất kỳ tham số nào được định nghĩa trong file experiment sẽ ghi đè lên giá trị tương ứng từ các file `base`.

**Ví dụ về `configs/v26/v26_m_demo.yaml`:**
```yaml
# Kế thừa từ 3 file base
base:
  - configs/_base_/datasets/coco_min.yaml
  - configs/_base_/schedules/default_10e.yaml
  - configs/_base_/models/yolo26m.yaml

# Ghi đè các tham số của lần chạy này
train:
  epochs: 5 # Chạy 5 epochs thay vì 10 từ schedule
  batch: 8
  name: "v26_m_demo_run" # Đổi tên thư mục kết quả
  amp: False
```
Cách tiếp cận này giúp giảm thiểu việc lặp lại code và giữ cho các cấu hình experiment luôn gọn gàng, dễ quản lý.
