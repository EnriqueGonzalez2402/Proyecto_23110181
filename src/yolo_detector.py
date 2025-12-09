from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

        # Palabras clave que consideraremos como "tanque"
        self.tank_keywords = [
            "tank", "armored", "armour", "military vehicle",
            "armored vehicle", "combat vehicle", "afv"
        ]

    def is_tank_label(self, label):
        label = label.lower()
        return any(keyword in label for keyword in self.tank_keywords)

    def detect_tanks(self, frame):
        results = self.model(frame, verbose=False)[0]

        detections = []

        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = results.names[cls_id]

            # Solo regresar si es tanque
            if not self.is_tank_label(label):
                continue

            # Coordenadas del bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(w), int(h)]
            })

        return detections
