import cv2
import json
from enemy import EnemyDetector

print("Sistema iniciado...")

# --- Cargar base de datos desde el JSON ---
with open("data\\tanks.json", "r", encoding="utf-8") as f:
    TANK_DATABASE = json.load(f)["tanks"]

# Función para buscar tanque por nombre
def buscar_tanque(nombre):
    for tanque in TANK_DATABASE:
        if tanque["name"].lower() == nombre.lower():
            return tanque
    return None

# EJEMPLO: tanque detectado por tu modelo (manual por ahora)
DETECTED_NAME = "MS-1"     # <-- CAMBIA ESTE NOMBRE
DETECTED_CLASS = "light"   # <-- light, medium, heavy, tank_destroyer

# Buscar la info en el JSON
info = buscar_tanque(DETECTED_NAME)

enemy = EnemyDetector()
cap = cv2.VideoCapture("videos/wot.mp4")

if not cap.isOpened():
    print("No se pudo abrir el video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    box, mask = enemy.detect_enemy(frame)

    if box is not None:
        x1, y1, x2, y2 = box

        # Dibujar recuadro
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)

        # Mostrar nombre + clase detectada
        cv2.putText(frame,
                    f"{DETECTED_NAME} - {DETECTED_CLASS}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,255,255), 2)

        # Si existe en el JSON, mostrar weakspots reales
        if info:
            weakspot = ", ".join(info["weakspots"])
            cv2.putText(frame,
                        f"Weakspots: {weakspot}",
                        (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,255), 2)

            cv2.putText(frame,
                        info["notes"],
                        (x1, y2 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)
        else:
            cv2.putText(frame,
                        "No hay datos en JSON",
                        (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,255), 2)

    cv2.imshow("Detector", frame)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
