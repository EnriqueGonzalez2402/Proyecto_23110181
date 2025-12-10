import cv2
import numpy as np

class EnemyDetector:
    def __init__(self):
        self.lower1 = np.array([0, 80, 80])
        self.upper1 = np.array([10, 255, 255])
        self.lower2 = np.array([170, 80, 80])
        self.upper2 = np.array([180, 255, 255])

    def detect_enemy(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self.lower1, self.upper1)
        mask2 = cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None, mask

        largest = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(largest)

        if w*h < 200:  
            return None, mask

        return (x, y, x+w, y+h), mask
