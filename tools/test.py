import yaml
import argparse
import os
from src.core.registry import MODELS
import src.modeling.yolo_wrapper 

def main():
    parser = argparse.ArgumentParser(description="Model Zoo Evaluation Script")
    parser.add_argument('--config', type=str, required=True, help="Model nickname (e.g. yolo11m)")
    args = parser.parse_args()

    # 1. Xác định file Zoo
    zoo_file = f"configs/_base_/models/{args.config[:4]}.yaml"
    
    if not os.path.exists(zoo_file):
        print(f"Error: Config file not found at {zoo_file}")
        return

    with open(zoo_file, 'r') as f:
        zoo_cfg = yaml.safe_load(f)

    # 2. Lấy thông tin model cụ thể
    if args.config not in zoo_cfg['models']:
        raise KeyError(f"Model {args.config} not found in Zoo.")
    
    model_cfg = zoo_cfg['models'][args.config]
    
    # Lấy thông số global nếu model không có
    imgsz = model_cfg.get('imgsz', zoo_cfg.get('common', {}).get('imgsz', 640))

    # 3. Khởi tạo model từ Registry
    print(f"Loading {args.config} (Type: {model_cfg['type']})...")
    model_class = MODELS.get(model_cfg['type'])
    model_instance = model_class(weights_path=model_cfg['weights_path'])

    # 4. Đánh giá (Evaluate)
    print(f"Evaluating on data: {zoo_cfg['dataset']['data_path']}")
    
    # Gọi hàm val() của Ultralytics thông qua wrapper
    # Lưu ý: data path nằm ở root của file yaml (zoo_cfg['dataset'])
    metrics = model_instance.model.val(
        data=zoo_cfg['dataset']['data_path'],
        imgsz=imgsz
    )
    
    print("-" * 30)
    print(f"Results for {args.config}:")
    print(f"mAP@50-95: {metrics.box.map}")
    print(f"mAP@50:    {metrics.box.map50}")

if __name__ == "__main__":
    main()