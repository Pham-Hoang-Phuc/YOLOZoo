# Model Card: YOLOv11m (Medium)

## 📌 Tổng quan (Overview)
- **Kiến trúc:** YOLOv11 Medium
- **Nhiệm vụ:** Object Detection (Phát hiện vật thể)
- **Đơn vị phát triển:** Ultralytics
- **Định dạng gốc:** PyTorch (.pt)

## ⚖️ Giấy phép (License)
- **Loại:** AGPL-3.0 (Strong Copyleft)
- **Lưu ý thương mại:** Yêu cầu mua bản quyền doanh nghiệp từ Ultralytics nếu sử dụng trong sản phẩm đóng (Closed Source) hoặc cung cấp dịch vụ SaaS mà không muốn công khai mã nguồn sản phẩm.

## 📊 Hiệu năng (Performance)
*Đo đạc trên tập dữ liệu COCO val2017:*
- **mAP@50-95:** ~52.7 (Tham khảo)
- **Kích thước đầu vào:** 640px
- **Độ trễ (Latency):** ~5.0ms (trên NVIDIA A100)

## ⚠️ Lưu ý khi sử dụng (Limitations)
1. **Dữ liệu huấn luyện:** Mô hình được huấn luyện trên COCO, có thể cần fine-tuning cho các bài toán đặc thù (như y tế, công nghiệp).
2. **Môi trường:** Yêu cầu thư viện `ultralytics >= 8.3.0`.
3. **Phần cứng:** Khuyến khích sử dụng GPU để đạt tốc độ thời gian thực.