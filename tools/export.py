import argparse
import yaml
import os
from ultralytics import YOLO

def export():
    parser = argparse.ArgumentParser(description="Model Zoo Export Script")
    # Thay vì nhập path weights, ta nhập tên model trong Zoo (vd: yolo11m)
    parser.add_argument('--config', type=str, required=True, help="Model nickname in Zoo")
    parser.add_argument('--format', type=str, default='onnx')
    args = parser.parse_args()

    # Logic xác định file Zoo (yolo11m -> configs/_base_/models/yolo.yaml)
    zoo_file = f"configs/_base_/models/{args.config[:4]}.yaml"
    
    if not os.path.exists(zoo_file):
        print(f"Error: Không tìm thấy file Zoo tại {zoo_file}")
        return

    with open(zoo_file, 'r') as f:
        zoo_cfg = yaml.safe_load(f)

    # Tìm model trong danh sách
    if args.config not in zoo_cfg['models']:
        print(f"Error: Model '{args.config}' không tồn tại trong {zoo_file}")
        return

    # Lấy đường dẫn trọng số từ config
    weights_path = zoo_cfg['models'][args.config]['weights_path']
    print(f"Found weights for {args.config}: {weights_path}")

    # Load model và export
    model = YOLO(weights_path)
    print(f"Exporting {args.config} to {args.format} format...")
    
    # Export (opset=11 là chuẩn ổn định cho ONNX)
    path = model.export(format=args.format, opset=11)
    print(f"Exported successfully to: {path}")

if __name__ == "__main__":
    export()