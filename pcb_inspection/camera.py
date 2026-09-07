import os
import time
import threading
import cv2
import numpy as np

from .config import (
    CAMERA_SOURCE,
    CAMERA_SENSOR_ID,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    FRAMERATE,
    TEST_IMAGE_PATH,
)


class Camera:
    """
    Multi-source low-latency camera interface supporting Jetson CSI (IMX477),
    V4L2 USB cameras, and Simulation fallback.
    Uses a dedicated background capture thread to drain camera buffers continuously,
    ensuring zero-lag real-time latest frame access (< 33ms latency).
    """

    def __init__(self, source: str = CAMERA_SOURCE, sensor_id: int = CAMERA_SENSOR_ID):
        self.requested_source = source
        self.sensor_id = sensor_id
        self.current_mode = "Disconnected"
        self.cap = None

        self._sim_frame = None
        self._last_sim_time = 0.0
        self._frame_interval = 1.0 / max(1, FRAMERATE)

        # Threaded capture components to eliminate buffer queuing latency
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame = None
        self._has_new_frame = False
        self._frame_count = 0
        self._capture_fps = 0.0

    @property
    def is_opened(self) -> bool:
        if self.current_mode.startswith("Simulation"):
            return self._sim_frame is not None
        return self.cap is not None and self.cap.isOpened()

    @property
    def capture_fps(self) -> float:
        return self._capture_fps

    def csi_pipeline(self) -> str:
        """GStreamer pipeline for IMX477 on Jetson CSI interface with low buffer latency."""
        return (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={CAMERA_WIDTH},height={CAMERA_HEIGHT},"
            f"format=NV12,framerate={FRAMERATE}/1 ! "
            "nvvidconv flip-method=0 ! "
            f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},format=BGRx ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

    def _start_capture_thread(self):
        """Start background capture worker that drains buffers continuously."""
        self._stop_capture_thread()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_worker,
            name="CameraCaptureThread",
            daemon=True,
        )
        self._thread.start()

    def _stop_capture_thread(self):
        """Stop background capture worker."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _capture_worker(self):
        """Continuously grab frames from camera to prevent hardware/driver queue bloat."""
        frame_counter = 0
        fps_timer = time.perf_counter()

        while not self._stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.01)
                continue

            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._has_new_frame = True
                    self._frame_count += 1
                frame_counter += 1

                now = time.perf_counter()
                elapsed = now - fps_timer
                if elapsed >= 1.0:
                    self._capture_fps = frame_counter / elapsed
                    frame_counter = 0
                    fps_timer = now
            else:
                time.sleep(0.002)

    def open(self, source: str = None) -> tuple:
        """
        Open the camera using requested source, falling back automatically if needed.
        Returns: (success: bool, mode: str, message: str)
        """
        self.release()
        target_source = source if source is not None else self.requested_source

        # 1. Try CSI Camera if requested or auto
        if target_source in ("auto", "csi"):
            try:
                pipeline = self.csi_pipeline()
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    self._start_capture_thread()
                    # Wait up to 2 seconds for first live frame
                    start_wait = time.perf_counter()
                    while time.perf_counter() - start_wait < 2.0:
                        with self._lock:
                            if self._latest_frame is not None:
                                self.current_mode = "CSI (IMX477)"
                                return True, self.current_mode, "Jetson CSI Camera connected."
                        time.sleep(0.05)
                self.release()
            except Exception as e:
                self.release()
                if target_source == "csi":
                    return False, "Error", f"CSI Camera failed: {e}"

        # 2. Try USB Camera if requested or auto
        if target_source in ("auto", "usb"):
            try:
                self.cap = cv2.VideoCapture(self.sensor_id)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    self.cap.set(cv2.CAP_PROP_FPS, FRAMERATE)
                    self._start_capture_thread()
                    start_wait = time.perf_counter()
                    while time.perf_counter() - start_wait < 2.0:
                        with self._lock:
                            if self._latest_frame is not None:
                                self.current_mode = f"USB (video{self.sensor_id})"
                                return True, self.current_mode, f"USB Camera {self.sensor_id} connected."
                        time.sleep(0.05)
                self.release()
            except Exception as e:
                self.release()
                if target_source == "usb":
                    return False, "Error", f"USB Camera failed: {e}"

        # 3. Fallback to Simulation Mode (test image)
        if target_source in ("auto", "simulation"):
            if os.path.exists(TEST_IMAGE_PATH):
                self._sim_frame = cv2.imread(TEST_IMAGE_PATH)
                if self._sim_frame is not None:
                    self.current_mode = "Simulation"
                    self._last_sim_time = time.perf_counter()
                    msg = f"Camera hardware offline. Running in Simulation mode ({os.path.basename(TEST_IMAGE_PATH)})."
                    return True, self.current_mode, msg

            # Synthetic PCB fallback image if test.jpg is missing
            self._sim_frame = np.full((1080, 1920, 3), (34, 139, 34), dtype=np.uint8)
            self.current_mode = "Simulation (Synthetic)"
            return True, self.current_mode, "Simulation active with synthetic image."

        self.current_mode = "Disconnected"
        return False, "Disconnected", f"Failed to initialize camera mode '{target_source}'."

    def read(self) -> tuple:
        """
        Read the latest available frame with zero latency.
        Returns: (ret: bool, frame: np.ndarray)
        """
        if self.current_mode.startswith("Simulation"):
            if self._sim_frame is None:
                return False, None

            # Pace simulated frames to target framerate
            now = time.perf_counter()
            elapsed = now - self._last_sim_time
            if elapsed < self._frame_interval:
                time.sleep(max(0.001, self._frame_interval - elapsed))
            self._last_sim_time = time.perf_counter()

            return True, self._sim_frame.copy()

        with self._lock:
            if self._latest_frame is not None:
                self._has_new_frame = False
                return True, self._latest_frame.copy()

        return False, None

    def get_latest_frame(self) -> tuple:
        """
        Returns (ret: bool, frame: np.ndarray, is_new: bool).
        Non-blocking and provides staleness flag.
        """
        if self.current_mode.startswith("Simulation"):
            ret, frame = self.read()
            return ret, frame, True

        with self._lock:
            if self._latest_frame is not None:
                is_new = self._has_new_frame
                self._has_new_frame = False
                return True, self._latest_frame.copy(), is_new
            return False, None, False

    def release(self):
        """Cleanly release camera resources and background thread."""
        self._stop_capture_thread()

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        with self._lock:
            self._latest_frame = None
            self._has_new_frame = False

        self._sim_frame = None
        self.current_mode = "Disconnected"
