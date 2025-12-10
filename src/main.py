import cv2
import json
import os
from screen_reader import ScreenReader
from yolo_detector import YOLODetector
from enemy_detector import EnemyColorDetector

# --- Cargar base de datos de tanques ---
JSON_PATH = os.path.join("data", "tanks.json")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    tank_database = json.load(f)

# --- Inicializar ---
screen = ScreenReader(monitor_index=1)
yolo = YOLODetector()
enemy = EnemyColorDetector()

print("Sistema iniciado...")

# --- Loop principal ---
while True:
    frame = screen.capture()

    # 1. detectar color rojo (enemigos)
    enemy_box, _ = enemy.detect_enemy(frame)

    # 2. si hay enemigo, usar YOLO para clasificar tanque
    if enemy_box:
        ex1, ey1, ex2, ey2 = enemy_box
        crop = frame[ey1:ey2, ex1:ex2]

        detections = yolo.detect_tanks(crop)

        if len(detections) > 0:
            best = max(detections, key=lambda x: x["confidence"])
            tank_name = best["label"]

            print("Tanque detectado:", tank_name)

            # Si el tanque existe en la base, imprimir datos
            if tank_name in tank_database:
                print("Stats:", tank_database[tank_name])
            else:
                print("Tanque no está en la base de datos.")

    cv2.imshow("WOT Detector", frame)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
