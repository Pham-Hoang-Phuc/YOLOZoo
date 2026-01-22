# tools/infer.py
import argparse
import os
import sys
from ultralytics import YOLO

# Import
sys.path.append(os.getcwd())
from src.core.config_parser import load_config
from src.core.data_manager import check_and_pull_data # <--- Import thêm hàm này

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    # weights ở đây là tùy chọn override từ dòng lệnh
    parser.add_argument('--weights', type=str, default=None) 
    args = parser.parse_args()

    # 1. Load Config
    cfg = load_config(args.config)

    # 2. Xác định đường dẫn weights
    # Ưu tiên 1: Dòng lệnh -> Ưu tiên 2: Config
    weights_path = args.weights if args.weights else cfg['model']['weights']
    
    # 3. Logic DVC cho Weights
    # Nếu dùng weights từ config, kiểm tra xem có file .dvc tương ứng không
    if not args.weights: 
        dvc_weights_file = cfg['model'].get('dvc_weights_file')
        if dvc_weights_file:
            # Tự động pull nếu file .pt chưa có
            check_and_pull_data(dvc_weights_file)

    # 4. Load Model
    print(f"--> Loading model from: {weights_path}")
    
    # Kiểm tra lần cuối xem file có tồn tại không để báo lỗi rõ ràng
    if not os.path.exists(weights_path) and not weights_path.endswith('.pt'):
        # Nếu path không tồn tại và không phải là alias (yolo11m.pt) thì báo lỗi
        print(f"Error: Weights file not found at {weights_path}")
        return

    model = YOLO(weights_path)

    # 5. Inference
    imgsz = cfg['train'].get('imgsz', 640)
    device = cfg['train'].get('device', 'cpu')

    results = model.predict(
        source=args.source,
        imgsz=imgsz,
        device=device,
        save=True,
        project="runs/detect",
        name="infer_result"
    )
    print(f"Done.")

if __name__ == "__main__":
    main()