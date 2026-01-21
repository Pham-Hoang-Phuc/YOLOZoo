from ultralytics import YOLO
from src.core.registry import MODELS

@MODELS.register_module
class YOLOv11Detector:
    def __init__(self, weights_path, task='detect'):
        self.model = YOLO(weights_path)
        self.task = task

    def predict(self, source, imgsz=640, conf=0.25):
        return self.model.predict(source, imgsz=imgsz, conf=conf)

    def train(self, data_config, epochs=50, imgsz=640):
        return self.model.train(data=data_config, epochs=epochs, imgsz=imgsz)