# tools/train.py
import argparse
import os
import sys

# Thêm thư mục hiện tại vào path để import được src
sys.path.append(os.getcwd())

from src.core.config_parser import load_config
from src.core.data_manager import check_and_pull_data
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLO Model Zoo Training")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to experiment config (e.g., configs/v11/v11_m_demo.yaml)')
    args = parser.parse_args()

    print(f"--> Loading config from: {args.config}")
    cfg = load_config(args.config)
    
    # 1. Chuẩn bị Dữ liệu (Auto DVC Pull)
    dataset_cfg = cfg.get('dataset', {})
    dvc_path = dataset_cfg.get('dvc_path')
    data_yaml_path = dataset_cfg.get('data_path')

    if dvc_path:
        success = check_and_pull_data(dvc_path)
        if not success:
            print("Stop training due to data error.")
            return

    # 2. Khởi tạo Model
    model_cfg = cfg.get('model', {})
    weights = model_cfg.get('weights', 'yolo11m.pt')
    
    print(f"--> Initializing model with weights: {weights}")
    model = YOLO(weights) 

    # 3. Training
    train_cfg = cfg.get('train', {})
    print(f"--> Starting training experiment: {train_cfg.get('name')}")
    
    # Lưu ý: Ultralytics cần đường dẫn tuyệt đối hoặc tương đối chuẩn tới data.yaml
    model.train(
        data=data_yaml_path,
        **train_cfg # Unpack các tham số: epochs, batch, imgsz, device...
    )

if __name__ == "__main__":
    main()