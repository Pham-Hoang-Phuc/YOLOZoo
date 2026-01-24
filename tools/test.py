import argparse
import os
import sys

# Thêm thư mục hiện tại vào path để import được src
sys.path.append(os.getcwd())

from ultralytics import YOLO
from src.core.config_parser import load_config
from src.core.data_manager import check_and_pull_data

def main():
    parser = argparse.ArgumentParser(description="Model Zoo Evaluation Script")
    parser.add_argument('--config', type=str, required=True, help="Path to the experiment config file (e.g., configs/v26/v26_m_demo.yaml)")
    parser.add_argument('--weights', type=str, default=None, help="Optional: Path to a specific weights file to override the one in the config.")
    args = parser.parse_args()

    # 1. Load Config
    print(f"--> Loading config from: {args.config}")
    cfg = load_config(args.config)

    # 2. Chuẩn bị Dữ liệu (Auto DVC Pull)
    dataset_cfg = cfg.get('dataset', {})
    dvc_path = dataset_cfg.get('dvc_path')
    data_yaml_path = dataset_cfg.get('data_path')

    if dvc_path:
        print("--> Checking dataset integrity...")
        if not check_and_pull_data(dvc_path):
            print("Error: Could not pull dataset. Aborting evaluation.")
            return
    
    # 3. Xác định và chuẩn bị Weights (Auto DVC Pull)
    model_cfg = cfg.get('model', {})
    # Ưu tiên 1: Dòng lệnh -> Ưu tiên 2: Config
    weights_path = args.weights if args.weights else model_cfg.get('weights')
    
    if not weights_path:
        print("Error: No weights specified in command line or config file. Aborting.")
        return

    # Nếu dùng weights từ config (không phải từ --weights), kiểm tra DVC
    if not args.weights:
        dvc_weights_file = model_cfg.get('dvc_weights_file')
        if dvc_weights_file:
            print("--> Checking model weights integrity...")
            if not check_and_pull_data(dvc_weights_file):
                print("Error: Could not pull model weights. Aborting evaluation.")
                return

    # 4. Khởi tạo Model
    print(f"--> Loading model from: {weights_path}")
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at '{weights_path}'. Please ensure the path is correct or run DVC pull.")
        return
        
    model = YOLO(weights_path)

    # 5. Đánh giá (Evaluate)
    print(f"--> Evaluating on data: {data_yaml_path}")
    
    # Lấy các tham số từ config, có giá trị mặc định
    train_cfg = cfg.get('train', {})
    imgsz = train_cfg.get('imgsz', 640)
    device = train_cfg.get('device', 'cpu')
    name = train_cfg.get('name', 'eval_run') + "_eval"

    metrics = model.val(
        data=data_yaml_path,
        imgsz=imgsz,
        device=device,
        name=name,
        exist_ok=True
    )
    
    print("-" * 40)
    print(f"Results for {weights_path}:")
    # In ra các chỉ số chính (cho cả detection và segmentation)
    if hasattr(metrics, 'box'):
        print(f"  mAP50-95(B): {metrics.box.map:.4f}")
        print(f"  mAP50(B):    {metrics.box.map50:.4f}")
    if hasattr(metrics, 'seg'):
        print(f"  mAP50-95(M): {metrics.seg.map:.4f}")
        print(f"  mAP50(M):    {metrics.seg.map50:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    main()