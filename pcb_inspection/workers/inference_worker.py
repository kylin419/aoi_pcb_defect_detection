import os
import time
import psutil
import numpy as np

from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot, QMetaObject, Qt

from ..camera import Camera
from ..detector import TensorRTDetector
from ..result_manager import ResultManager
from ..visualization import draw_detections
from ..ros.publisher import AoiRosNode
from ..config import (
    AUTO_SAVE_NG,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    CLASSES,
)


class FpsTracker:
    """Windowed moving average FPS tracker for stable, accurate FPS measurement."""

    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.timestamps = []
        self._fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

        if len(self.timestamps) >= 2:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            if elapsed > 0:
                self._fps = (len(self.timestamps) - 1) / elapsed
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps

    def reset(self):
        self.timestamps.clear()
        self._fps = 0.0


class InferenceWorker(QObject):
    """
    Background worker orchestrating:
    - Camera acquisition (CSI / USB / Simulation)
    - YOLOv12 TensorRT defect detection
    - ROS 2 image, result, and status publishing
    - Production yield metrics & hardware telemetry
    - Automated saving and snapshot management
    """

    frame_ready = pyqtSignal(object)
    dashboard_ready = pyqtSignal(dict)
    status_ready = pyqtSignal(dict)
    fps_ready = pyqtSignal(float)

    log = pyqtSignal(str)
    error = pyqtSignal(str)

    started = pyqtSignal()
    stopped = pyqtSignal()
    ros_command_received = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.camera = None
        self.detector = None
        self.ros_node = None
        self.timer = None

        self.running = False
        self.initialized = False
        self.fps_tracker = FpsTracker(window_size=15)
        self._last_preview_dash_time = 0.0

        self.result_manager = ResultManager()

        # Production yield statistics
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.auto_save_ng = AUTO_SAVE_NG

        # Frame caches
        self.last_frame = None
        self.last_overlay = None
        self.last_result = None

        # Hardware monitor cache & timer throttle
        self.last_hw_time = 0.0
        self.hw_stats = {"gpu": "--", "temperature": "--", "memory": "--"}

    @pyqtSlot()
    def initialize(self):
        """Initialize camera, TensorRT engine, and ROS 2 node."""
        try:
            self.log.emit("Initializing Camera...")
            self.camera = Camera()
            success, mode, cam_msg = self.camera.open()
            self.log.emit(f"Camera [{mode}]: {cam_msg}")

            self.log.emit("Loading TensorRT YOLOv12 Engine...")
            self.detector = TensorRTDetector()
            self.log.emit("TensorRT Engine Ready.")

            self.log.emit("Starting ROS 2 Node...")
            self.ros_node = AoiRosNode()
            self.ros_node.set_command_callback(self._on_ros_command)
            self.log.emit("ROS 2 Interface Ready.")

            # Frame polling timer (checks at ~60 Hz for fresh frames)
            self.timer = QTimer(self)
            self.timer.setInterval(15)
            self.timer.timeout.connect(self.process_frame)
            self.timer.start()

            self.initialized = True
            self.log.emit("AOI Inspection Pipeline Fully Initialized (Live Preview Active).")

            # Emit initial ready status
            self.status_ready.emit({
                "camera": True,
                "camera_mode": mode,
                "engine": True,
                "ros": True,
            })

        except Exception as e:
            self.error.emit(f"Initialization error: {e}")

    def _on_ros_command(self, cmd: str):
        """Dispatch ROS 2 external command safely to worker QThread."""
        self.ros_command_received.emit(cmd)
        cmd_map = {
            "START": "start",
            "STOP": "stop",
            "TRIGGER": "trigger",
            "SNAPSHOT": "snapshot",
            "SAVE": "save_result",
            "RESET_STATS": "reset_statistics",
        }
        method = cmd_map.get(cmd)
        if method:
            QMetaObject.invokeMethod(
                self,
                method,
                Qt.ConnectionType.QueuedConnection,
            )

    def _read_hardware_telemetry(self) -> dict:
        """Query Jetson SoC temperature, GPU load, and system memory."""
        now = time.perf_counter()
        if now - self.last_hw_time < 1.0:
            return self.hw_stats
        self.last_hw_time = now

        # Temperature
        temp_str = "--"
        for p in (
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/devices/virtual/thermal/thermal_zone0/temp",
        ):
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        val = float(f.read().strip())
                        temp_str = f"{val/1000.0:.1f} °C" if val > 1000 else f"{val:.1f} °C"
                        break
                except Exception:
                    pass

        # GPU Load
        gpu_str = "--"
        for p in (
            "/sys/devices/gpu.0/load",
            "/sys/devices/platform/17000000.ga10b/load",
        ):
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        val = float(f.read().strip())
                        gpu_str = f"{val/10.0:.0f}%" if val <= 1000 else f"{val:.0f}%"
                        break
                except Exception:
                    pass

        # Memory
        mem_str = "--"
        try:
            mem = psutil.virtual_memory()
            mem_str = f"{mem.used / (1024**3):.1f} / {mem.total / (1024**3):.1f} GB"
        except Exception:
            pass

        self.hw_stats = {
            "temperature": temp_str,
            "gpu": gpu_str,
            "memory": mem_str,
        }
        return self.hw_stats

    def _build_dashboard(self, fps: float, result: dict) -> dict:
        """Construct dashboard data dictionary."""
        yield_rate = (
            (self.ok_count / self.total_count * 100.0)
            if self.total_count > 0
            else 100.0
        )

        dashboard = {
            "fps": f"{fps:.1f}",
            "inference": f"{result.get('inference_ms', 0.0):.1f} ms",
            "camera": self.camera.current_mode if self.camera else "Disconnected",
            "model": "YOLOv12 (TRT)",
            "total_count": str(self.total_count),
            "ok_count": str(self.ok_count),
            "ng_count": str(self.ng_count),
            "yield_rate": f"{yield_rate:.1f}%",
        }

        # Defect counts
        stats = result.get("stats", {})
        for c in CLASSES:
            dashboard[c] = stats.get(c, 0)
        # Compatibility mapping for dashboard card 'missing'
        if "missing_hole" in stats:
            dashboard["missing"] = stats["missing_hole"]

        dashboard.update(self._read_hardware_telemetry())
        return dashboard

    @pyqtSlot()
    def start(self):
        """Start continuous inspection."""
        if not self.initialized:
            self.error.emit("Worker is not initialized.")
            return

        if self.running:
            return

        self.running = True
        self.fps_tracker.reset()
        self.started.emit()
        self.log.emit("Continuous Defect Inspection Started.")

    @pyqtSlot()
    def stop(self):
        """Stop continuous inspection (live preview remains active)."""
        if not self.running:
            return

        self.running = False
        self.fps_tracker.reset()
        self.stopped.emit()
        self.log.emit("Continuous Inspection Paused (Live Camera Active).")

    @pyqtSlot()
    def trigger(self):
        """Single-shot inspection (e.g. triggered by sensor or button)."""
        if not self.initialized:
            return
        self.process_frame(is_trigger=True)

    @pyqtSlot()
    def process_frame(self, is_trigger: bool = False):
        """Acquire frame, infer defects (if running/trigger), publish to ROS 2, and update UI."""
        if not self.initialized or self.camera is None:
            return

        try:
            # Query the freshest frame from camera capture thread
            ret, frame, is_new = self.camera.get_latest_frame()

            if not ret or frame is None:
                return

            # Avoid redundant processing if a new frame hasn't arrived yet
            if not is_trigger and not is_new:
                return

            # Update smoothed FPS
            fps = self.fps_tracker.update()
            if fps <= 0.0 and hasattr(self.camera, "capture_fps") and self.camera.capture_fps > 0:
                fps = self.camera.capture_fps

            # -------------------------------------------------------------
            # 1. LIVE PREVIEW MODE (Standby when inspection is not started)
            # -------------------------------------------------------------
            if not self.running and not is_trigger:
                self.last_frame = frame
                self.frame_ready.emit(frame)
                self.fps_ready.emit(fps)

                now = time.perf_counter()
                if now - self._last_preview_dash_time >= 0.5:
                    self._last_preview_dash_time = now
                    preview_dash = self._build_dashboard(fps, {"inference_ms": 0.0, "ok": True})
                    self.dashboard_ready.emit(preview_dash)
                return

            # -------------------------------------------------------------
            # 2. DEFECT INSPECTION MODE (Continuous or Single Trigger)
            # -------------------------------------------------------------
            result = self.detector.detect(frame)

            # Draw visual defect overlay
            overlay = draw_detections(
                frame,
                result["detections"],
                inference_ms=result["inference_ms"],
            )

            # Update production yield counters
            self.total_count += 1
            if result["ok"]:
                self.ok_count += 1
            else:
                self.ng_count += 1

            self.last_frame = frame
            self.last_overlay = overlay
            self.last_result = result

            # Publish to ROS 2 (throttled)
            if self.ros_node is not None:
                self.ros_node.publish_inspection(
                    raw_frame=frame,
                    annotated_frame=overlay,
                    result=result,
                    fps=fps,
                )

            # Auto-save NG if enabled
            if self.auto_save_ng and not result["ok"]:
                folder = self.result_manager.save_all(frame, overlay, result)
                self.log.emit(f"Auto-saved NG Defect: {folder}")

            # Emit GUI signals
            self.frame_ready.emit(overlay)
            self.fps_ready.emit(fps)
            self.dashboard_ready.emit(self._build_dashboard(fps, result))

        except Exception as e:
            self.error.emit(f"Processing error: {e}")

    @pyqtSlot()
    def snapshot(self):
        """Save current raw frame."""
        if self.last_frame is None:
            self.log.emit("No frame to save.")
            return

        folder = self.result_manager.create_folder()
        self.result_manager.save_image(folder, self.last_frame)
        self.log.emit(f"Snapshot saved: {folder}")

    @pyqtSlot()
    def save_result(self):
        """Save current raw frame, overlay, and detection JSON."""
        if self.last_result is None or self.last_frame is None:
            self.log.emit("No result to save.")
            return

        folder = self.result_manager.save_all(
            self.last_frame,
            self.last_overlay,
            self.last_result,
        )
        self.log.emit(f"Inspection Result saved: {folder}")

    @pyqtSlot(float, float)
    def set_thresholds(self, confidence: float, iou: float):
        """Update detection thresholds dynamically."""
        if self.detector is not None:
            self.detector.set_thresholds(confidence, iou)
            self.log.emit(f"Updated thresholds: Conf={confidence:.2f}, IoU={iou:.2f}")

    @pyqtSlot(str)
    def set_camera_source(self, source: str):
        """Switch camera input source."""
        if self.camera is not None:
            was_running = self.running
            if was_running:
                self.stop()
            success, mode, msg = self.camera.open(source)
            self.fps_tracker.reset()
            self.log.emit(f"Camera switched to [{mode}]: {msg}")
            self.status_ready.emit({
                "camera": success,
                "camera_mode": mode,
            })
            if was_running:
                self.start()

    @pyqtSlot(bool)
    def set_auto_save_ng(self, enabled: bool):
        """Toggle auto-saving of NG defects."""
        self.auto_save_ng = enabled
        self.log.emit(f"Auto-save NG: {'Enabled' if enabled else 'Disabled'}")

    @pyqtSlot()
    def reset_statistics(self):
        """Reset total, OK, and NG counts."""
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.log.emit("Production yield statistics reset.")
        if self.last_result:
            self.dashboard_ready.emit(self._build_dashboard(0.0, self.last_result))

    @pyqtSlot()
    def reload_detector(self):
        """Reload TensorRT detector engine."""
        try:
            self.log.emit("Reloading TensorRT Engine...")
            if self.detector is not None:
                self.detector.close()
            self.detector = TensorRTDetector()
            self.log.emit("TensorRT Engine reloaded successfully.")
        except Exception as e:
            self.error.emit(f"Reload engine failed: {e}")

    @pyqtSlot()
    def shutdown(self):
        """Clean up all worker resources, camera, ROS node, and CUDA."""
        self.running = False
        self.initialized = False

        try:
            if self.timer:
                self.timer.stop()

            if self.camera:
                self.camera.release()

            if self.ros_node:
                self.ros_node.close()

            if self.detector:
                self.detector.close()

            self.log.emit("Inference Worker cleanly shut down.")

        except Exception as e:
            self.error.emit(f"Shutdown error: {e}")

