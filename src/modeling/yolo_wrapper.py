from ultralytics import YOLO
from src.core.registry import MODELS

@MODELS.register_module
class YOLOv11Detector:
    def __init__(self, weights_path):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)

    # Thêm **kwargs vào đây để nhận iou hoặc bất kỳ tham số nào khác từ YAML
    def predict(self, source, imgsz=640, conf=0.25, **kwargs):
        return self.model.predict(
            source=source, 
            imgsz=imgsz, 
            conf=conf, 
            **kwargs  # Truyền tiếp xuống model gốc của Ultralytics
        )