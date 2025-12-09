import cv2
from screen_reader import ScreenReader
from yolo_detector import YOLODetector
from tank_database import TankDatabase

# Inicialización
reader = ScreenReader(monitor_index=1)
detector = YOLODetector()
db = TankDatabase()

while True:
    frame = reader.capture()
    detections = detector.detect_tanks(frame)

    for det in detections:
        if not det["enemy"]:
            continue  # solo enemigos

        tank_info = db.find_tank(det["label"])

        x1, y1, x2, y2 = map(int, det["box"].xyxy[0])

        # Dibujar caja roja
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Texto con nombre detectado
        cv2.putText(frame, det["label"], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        if tank_info:
            text = f"{tank_info['name']} | HP: {tank_info['hp']} | Weak: {', '.join(tank_info['weakspots'])}"
        else:
            text = f"{det['label']} (sin datos)"

        cv2.putText(frame, text, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("Tank Assistant", frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
