import argparse
from ultralytics import YOLO

def export():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--format', type=str, default='onnx')
    args = parser.parse_args()

    model = YOLO(args.weights)
    print(f"Exporting {args.weights} to {args.format} format...")
    
    # Xuất khẩu sang ONNX với opset=11 (chuẩn phổ biến)
    path = model.export(format=args.format, opset=11)
    print(f"Exported successfully to: {path}")

if __name__ == "__main__":
    export()