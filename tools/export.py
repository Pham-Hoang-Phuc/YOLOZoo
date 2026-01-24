import argparse
import os
import sys

# Thêm thư mục hiện tại vào path để import được src
sys.path.append(os.getcwd())

from ultralytics import YOLO
from src.core.config_parser import load_config
from src.core.data_manager import check_and_pull_data

def main():
    parser = argparse.ArgumentParser(description="Model Zoo Export Script")
    parser.add_argument('--config', type=str, required=True, help="Path to the experiment config file (e.g., configs/v26/v26_m_demo.yaml)")
    parser.add_argument('--weights', type=str, default=None, help="Optional: Path to a specific weights file to override the one in the config.")
    parser.add_argument('--format', type=str, default='onnx', help="Format to export to (e.g., onnx, engine, tflite)")
    args = parser.parse_args()

    # 1. Load Config
    print(f"--> Loading config from: {args.config}")
    cfg = load_config(args.config)
    
    # 2. Xác định và chuẩn bị Weights (Auto DVC Pull)
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
                print("Error: Could not pull model weights. Aborting export.")
                return

    # 3. Khởi tạo Model
    print(f"--> Loading model from: {weights_path}")
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at '{weights_path}'. Please ensure the path is correct or run DVC pull.")
        return
        
    model = YOLO(weights_path)

    # 4. Export
    # Lấy các tham số từ config, có giá trị mặc định
    train_cfg = cfg.get('train', {})
    imgsz = train_cfg.get('imgsz', 640)
    
    print(f"--> Exporting model to {args.format.upper()} format with image size {imgsz}...")
    
    try:
        # Lấy một số tham số export phổ biến từ config nếu có
        export_params = cfg.get('export', {})
        opset = export_params.get('opset', 12) # Mặc định opset 12 cho ONNX

        path = model.export(
            format=args.format,
            imgsz=imgsz,
            opset=opset,
            **export_params # Truyền các tham số khác như half, int8, etc.
        )
        print(f"--> Exported successfully to: {path}")
    except Exception as e:
        print(f"--> An error occurred during export: {e}")

if __name__ == "__main__":
    main()