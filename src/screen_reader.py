import cv2
import mss
import numpy as np

class ScreenReader:
    def __init__(self, monitor_index=1):
        """
        ScreenReader captura la pantalla desde un monitor específico.
        monitor_index=1 → típicamente tu segundo monitor.
        """
        self.sct = mss.mss()
        self.monitor_index = monitor_index

        if monitor_index >= len(self.sct.monitors):
            raise ValueError(f"Monitor {monitor_index} no encontrado. Disponibles: {len(self.sct.monitors)-1}")

        self.monitor = self.sct.monitors[monitor_index]

    def capture(self):
        """
        Captura el frame actual del monitor seleccionado.
        """
        img = np.array(self.sct.grab(self.monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


# === Test ===
if __name__ == "__main__":
    reader = ScreenReader(monitor_index=1)
    while True:
        frame = reader.capture()
        cv2.imshow("Screen Monitor #1", frame)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()

from yolo_detector import YOLODetector

detector = YOLODetector()

def process_frame(frame):
    detections = detector.detect_tanks(frame)

    if len(detections) > 0:
        best = max(detections, key=lambda x: x["confidence"])
        return best["label"]

    return None