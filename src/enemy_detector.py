import cv2
import numpy as np

class EnemyDetector:
    def __init__(self):
        # rangos para detectar color rojo en pantalla
        self.lower1 = np.array([0, 100, 100])
        self.upper1 = np.array([10, 255, 255])
        self.lower2 = np.array([160, 100, 100])
        self.upper2 = np.array([180, 255, 255])

    def get_enemy_regions(self, frame):
        """
        Devuelve áreas donde hay texto o elementos ROJOS típicos del WOT.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower1, self.upper1)
        mask2 = cv2.inRange(hsv, self.lower2, self.upper2)
        mask = mask1 | mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 30 and h > 10:  # evitar ruido
                regions.append((x, y, x+w, y+h))

        return regions

