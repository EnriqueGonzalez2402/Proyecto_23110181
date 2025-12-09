from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect_tanks(self, frame):
        results = self.model(frame, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = results.names[cls_id]
            detections.append({
                "label": label,
                "confidence": conf
            })

        return detections
