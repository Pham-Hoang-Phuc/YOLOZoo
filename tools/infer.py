import yaml
import argparse
import cv2
from src.core.registry import MODELS
# Import để trigger decorator đăng ký model
import src.modeling.yolo_wrapper 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    args = parser.parse_args()

    # 1. Đọc config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. Khởi tạo model từ Registry (Registry Pattern)
    model_class = MODELS.get(cfg['model']['type'])
    model_instance = model_class(weights_path=cfg['model']['weights_path'])

    # 3. Chạy Inference
    results = model_instance.predict(
        source=args.source, 
        imgsz=cfg['model']['imgsz'],
        conf=cfg['model']['conf_threshold']
    )

    # 4. Hiển thị kết quả
    for r in results:
        im_array = r.plot() 
        cv2.imshow("Model Zoo Demo", im_array)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()