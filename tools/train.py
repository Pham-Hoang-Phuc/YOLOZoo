import yaml
import argparse
import os
from src.core.registry import MODELS
import src.modeling.yolo_wrapper 

def main():
    parser = argparse.ArgumentParser(description="Model Zoo Training Script")
    parser.add_argument('--config', type=str, required=True, help='Model nickname (e.g. yolo8m)')
    args = parser.parse_args()

    # 1. Load cấu hình Zoo
    zoo_file = f"configs/_base_/models/{args.config[:4]}.yaml"
    
    if not os.path.exists(zoo_file):
        print(f"Error: Config file not found at {zoo_file}")
        return

    with open(zoo_file, 'r') as f:
        zoo_cfg = yaml.safe_load(f)

    # 2. Lấy config của model và dataset
    if args.config not in zoo_cfg['models']:
        raise KeyError(f"Model {args.config} not found in Zoo.")

    model_cfg = zoo_cfg['models'][args.config]
    dataset_cfg = zoo_cfg['dataset'] # Lấy block dataset ở ngoài cùng

    # Fallback logic cho các tham số
    imgsz = model_cfg.get('imgsz', zoo_cfg.get('common', {}).get('imgsz', 640))
    epochs = dataset_cfg.get('epochs', 10)
    batch_size = dataset_cfg.get('batch_size', 16)

    # 3. Khởi tạo model qua Registry
    print(f"[*] Initializing {args.config} for training...")
    model_class = MODELS.get(model_cfg['type'])
    
    # Load weights (có thể là pretrained hoặc checkpoint cũ)
    model_instance = model_class(weights_path=model_cfg['weights_path'])

    # 4. Kích hoạt training
    # Truyền các tham số đã parse được vào hàm train của wrapper
    model_instance.train(
        data=dataset_cfg['data_path'], # Lưu ý: tham số trong YOLO là 'data', không phải 'data_config'
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=zoo_cfg.get('common', {}).get('device', 'cuda')
    )

if __name__ == "__main__":
    main()