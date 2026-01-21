import yaml
import argparse
import cv2
import os
from src.core.registry import MODELS
import src.modeling.yolo_wrapper 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    args = parser.parse_args()

    ZOO_CONFIG_PATH = "configs/_base_/models/yolo.yaml"

    if not os.path.exists(ZOO_CONFIG_PATH):
        print(f"Error: Không tìm thấy file cấu hình {ZOO_CONFIG_PATH}")
        return

    # 1. Đọc file Zoo
    with open(ZOO_CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)

    # Kiểm tra xem nickname có tồn tại trong file không
    model_nickname = args.config
    if model_nickname not in cfg['models']:
        print(f"Error: Model '{model_nickname}' không tồn tại trong {ZOO_CONFIG_PATH}")
        return

    # Lấy thông tin model cụ thể
    selected_model_cfg = cfg['models'][model_nickname]

    # 2. Khởi tạo model từ Registry
    # Lấy class name từ key 'type'
    model_type = selected_model_cfg['type']
    model_class = MODELS.get(model_type)
    
    if model_class is None:
        print(f"Error: Class '{model_type}' chưa được đăng ký trong Registry.")
        return

    model_instance = model_class(weights_path=selected_model_cfg['weights_path'])

    # 3. Chạy Inference với logic Fallback tham số
    # Ưu tiên: Thông số trong model > Thông số trong common > Giá trị mặc định
    imgsz = selected_model_cfg.get('imgsz', cfg.get('common', {}).get('imgsz', 640))
    conf = selected_model_cfg.get('conf_threshold', cfg.get('common', {}).get('conf_threshold', 0.25))
    
    # Lấy các tham số bổ sung nếu có (như iou)
    extra_params = selected_model_cfg.get('params', {})

    print(f"Running inference: {model_nickname} | Type: {model_type} | Imgsz: {imgsz}")

    results = model_instance.predict(
        source=args.source, 
        imgsz=imgsz,
        conf=conf,
        **extra_params
    )

    # 4. Hiển thị kết quả
    for r in results:
        im_array = r.plot() 
        cv2.imshow(f"ViAIM Zoo - {model_nickname}", im_array)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()