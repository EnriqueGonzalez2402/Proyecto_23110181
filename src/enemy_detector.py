import cv2
import numpy as np

class EnemyColorDetector:
    def __init__(self):
        # Rango de ROJO en HSV (ajustable)
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])

    def detect_enemy(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Limpiar ruido
        mask = cv2.medianBlur(mask, 5)

        # Buscar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None, mask

        # Tomar el contorno más grande (el tanque enemigo)
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        return (x, y, w, h), mask


# === TEST DIRECTO ===
if __name__ == "__main__":
    img = cv2.imread("data\enemy_test.png")  # pon tu screenshot aquí
    detector = EnemyColorDetector()

    box, mask = detector.detect_enemy(img)

    if box:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Mask rojo", mask)
    cv2.imshow("Detección", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

