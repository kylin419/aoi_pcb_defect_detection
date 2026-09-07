import time

from .camera import Camera
from .detector import TensorRTDetector
from .visualization import draw_detections
from . import state


def inference_loop():
    """Main inference loop for standalone or web streaming."""
    cam = Camera()
    cam.open()
    detector = TensorRTDetector()

    prev_time = time.perf_counter()
    try:
        while True:
            if not state.running:
                time.sleep(0.1)
                continue

            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            result = detector.detect(frame)
            overlay = draw_detections(
                frame,
                result["detections"],
                inference_ms=result["inference_ms"],
            )

            now = time.perf_counter()
            fps = 0.0 if now == prev_time else 1.0 / (now - prev_time)
            prev_time = now

            state.frame = overlay
            state.stats = result["stats"]
            state.fps = fps

    finally:
        detector.close()
        cam.release()