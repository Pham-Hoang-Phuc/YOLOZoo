import yaml
import argparse
from src.core.registry import MODELS
import src.modeling.yolo_wrapper 

def main():
    parser = argparse.ArgumentParser(description="Model Zoo Evaluation Script")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Khởi tạo model từ Registry
    model_class = MODELS.get(cfg['model']['type'])
    model_instance = model_class(weights_path=cfg['model']['weights_path'])

    print("Evaluating model performance...")
    # Gọi hàm evaluate (YOLO của Ultralytics hỗ trợ sẵn hàm .val())
    metrics = model_instance.model.val(data=cfg['dataset']['data_path'])
    
    print(f"mAP@50-95: {metrics.box.map}")
    print(f"mAP@50: {metrics.box.map50}")

if __name__ == "__main__":
    main()  