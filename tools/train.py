import yaml
import argparse
from src.core.registry import MODELS
import src.modeling.yolo_wrapper # Cần thiết để đăng ký model vào Registry

def main():
    parser = argparse.ArgumentParser(description="Model Zoo Training Script")
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config')
    args = parser.parse_args()

    # 1. Load cấu hình từ file YAML
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. Khởi tạo mô hình thông qua Registry
    # Cơ chế này giúp bạn đổi từ YOLO sang model khác chỉ bằng cách đổi file config
    model_class = MODELS.get(cfg['model']['type'])
    model_instance = model_class(weights_path=cfg['model']['weights_path'])

    print(f"Starting training for {cfg['model']['name']}...")

    # 3. Kích hoạt quy trình huấn luyện
    model_instance.train(
        data_config=cfg['dataset']['data_path'],
        epochs=cfg['dataset']['epochs'],
        imgsz=cfg['model']['imgsz']
    )

if __name__ == "__main__":
    main()