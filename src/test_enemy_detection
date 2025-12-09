import cv2
from screen_reader import ScreenReader
from enemy_detector import EnemyDetector

reader = ScreenReader(monitor_index=1)
enemy = EnemyDetector()

while True:
    frame = reader.capture()

    regions = enemy.get_enemy_regions(frame)
    for (x1, y1, x2, y2) in regions:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("Enemy Detection", frame)
    if cv2.waitKey(1) == 27:
        break
    
from yolo_detector import YOLODetector

detector = YOLODetector()

def process_frame(frame):
    detections = detector.detect_tanks(frame)

    if len(detections) > 0:
        best = max(detections, key=lambda x: x["confidence"])
        return best["label"]

    return None

cv2.destroyAllWindows()
