import os
import time
from collections import deque
import psutil
import numpy as np

from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot, QMetaObject, Qt

from ..camera import Camera
from ..detector import TensorRTDetector
from ..result_manager import ResultManager
from ..visualization import draw_detections
from ..ros.publisher import AoiRosNode
from ..serial_bridge import ArduinoBridge
from ..config import (
    AUTO_SAVE_NG,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    CLASSES,
)
from .localizer import pcb_green_area_detection

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
    inspection_state = pyqtSignal(str)
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
        self.serial_bridge = None
        self.timer = None

        self.running = False
        
        self.initialized = False
        self.camera_ready = False
        
        self.fps_tracker = FpsTracker(window_size=15)
        self._last_preview_dash_time = 0.0
        # pcb state
        self.pcb_stop_sent=False
        self.inspect_waiting=False
        self.pcb_present=False
        self._stop_fault=False
        self._ng_hold=False
        self._ng_removed_at=None
        self._ng_absence_votes=deque(maxlen=5)
        self._pcb_detection_votes = deque(maxlen=5)

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
        """Initialize camera, TensorRT engine, ROS 2 node, and Arduino Due serial bridge."""
        try:
            self.log.emit("Initializing Camera...")
            self.camera = Camera()
            success, mode, cam_msg = self.camera.open()
            self.camera_ready = success
            self.log.emit(f"Camera [{mode}]: {cam_msg}")

            self.log.emit("Loading TensorRT YOLOv12 Engine...")
            self.detector = TensorRTDetector()
            self.log.emit("TensorRT Engine Ready.")

            self.log.emit("Starting ROS 2 Node...")
            self.ros_node = AoiRosNode()
            self.ros_node.set_command_callback(self._on_ros_command)
            self.log.emit("ROS 2 Interface Ready.")

            self.log.emit("Connecting to Arduino Due Serial Bridge...")
            self.serial_bridge = ArduinoBridge(
                feedback_callback=lambda msg: self.log.emit(f"Arduino: {msg}"),
            )
            if self.serial_bridge.connect():
                self.log.emit("Arduino Due Serial Bridge Connected.")
            else:
                self.log.emit("Arduino Due Serial Port not connected (Running in Standalone mode).")

            # Frame polling timer (checks at ~60 Hz for fresh frames)
            self.timer = QTimer(self)
            self.timer.setInterval(15)
            self.timer.timeout.connect(self.process_frame)
            self.timer.start()

            self.initialized = True
            self.log.emit("AOI Inspection Pipeline Fully Initialized (Live Preview Active).")

            # Emit initial ready status
            self.status_ready.emit({
                "camera": success,
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

        if not self.camera_ready:
            self.inspection_state.emit("ERROR")
            self.error.emit("Cannot start conveyor inspection: camera is not available.")
            return

        if self.running:
            return

        self.running = True
        # Reset PCB inspection state
        self.pcb_stop_sent=False
        self.inspect_waiting=False
        self.pcb_present=False
        self._stop_fault=False
        self._ng_hold=False
        self._ng_removed_at=None
        self._ng_absence_votes.clear()
        self._pcb_detection_votes.clear()
        self.fps_tracker.reset()
        if self.serial_bridge:
            self.serial_bridge.send_start()
        self.started.emit()
        self.inspection_state.emit("SEARCHING")
        self.log.emit("Continuous Defect Inspection Started.")

    @pyqtSlot()
    def stop(self):
        """Stop continuous inspection (live preview remains active)."""
        if not self.running:
            return

        self.running = False
        self.pcb_stop_sent=False
        self.inspect_waiting=False
        self.fps_tracker.reset()
        if self.serial_bridge:
            self.serial_bridge.send_stop()
        self.inspection_state.emit("IDLE")
        self.stopped.emit()
        self.log.emit("Continuous Inspection Paused (Live Camera Active).")

    @pyqtSlot()
    def trigger(self):
        """Run a single-shot inspection from the GUI or ROS command."""
        if not self.initialized:
            return
        self.inspection_state.emit("INSPECTING")
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
            # Camera is the sole automatic PCB arrival source. A 3-of-5 vote
            # rejects isolated green frames while admitting partly visible boards.
            if self.running and is_new:
                green_ratio = pcb_green_area_detection(frame)
                if not self.pcb_present:
                    self._pcb_detection_votes.append(green_ratio >= 0.08)
                    if sum(self._pcb_detection_votes) >= 3:
                        self.inspection_state.emit("PCB_DETECTED")
                        self.log.emit(f"PCB detected by camera (green area {green_ratio:.3f}).")
                        if self.serial_bridge and self.serial_bridge.send_ready():
                            self.pcb_stop_sent = True
                            self.inspect_waiting = True
                            self.pcb_present = True
                            self._pcb_detection_votes.clear()
                            self.last_frame = frame
                            self.frame_ready.emit(frame)
                            self.inspection_state.emit("STOP_CONFIRMED")
                            self.log.emit("Arduino confirmed conveyor stopped; starting defect inspection.")
                            return
                        elif self.serial_bridge:
                            self.pcb_present = True
                            self._stop_fault = True
                            self._pcb_detection_votes.clear()
                            self.inspection_state.emit("STOP_UNCONFIRMED")
                            self.log.emit("PCB detected, but Arduino stop was not confirmed; inspect the controller connection and motor state before restarting.")
                elif not self.inspect_waiting and not self._stop_fault:
                    if self._ng_hold:
                        self._ng_absence_votes.append(green_ratio < 0.025)
                        if sum(self._ng_absence_votes) >= 3:
                            if self._ng_removed_at is None:
                                self._ng_removed_at = time.monotonic()
                                self.inspection_state.emit("NG_RESUME_DELAY")
                        else:
                            self._ng_removed_at = None
                            self.inspection_state.emit("NG_WAIT_REMOVE")
                    elif green_ratio < 0.025:
                        self.pcb_present = False
                        self._pcb_detection_votes.clear()
                        self.inspection_state.emit("SEARCHING")
                        self.log.emit("PCB left camera view. Ready for next PCB.")

            if (
                self.running
                and self._ng_hold
                and not self._stop_fault
                and self._ng_removed_at is not None
                and time.monotonic() - self._ng_removed_at >= 2.0
                and sum(self._ng_absence_votes) >= 3
            ):
                if self.serial_bridge and self.serial_bridge.send_resume():
                    self._ng_hold = False
                    self._ng_removed_at = None
                    self._ng_absence_votes.clear()
                    self.pcb_present = False
                    self._pcb_detection_votes.clear()
                    self.inspection_state.emit("SEARCHING")
                    self.log.emit("NG PCB removal confirmed; Arduino resumed the conveyor after a 2-second delay.")
                else:
                    self._stop_fault = True
                    self.inspection_state.emit("RESUME_UNCONFIRMED")
                    self.log.emit("Arduino did not confirm conveyor resume; the system remains stopped.")

            # Continue into defect inference after camera detection; only hold
            # subsequent frames once inspection has completed.
            if self.running and self.pcb_present and not self.inspect_waiting:
                return

            # While the conveyor is searching, keep the live preview moving
            # without running defect inference on empty frames.
            if self.running and self.serial_bridge and not self.pcb_present:
                if not is_new:
                    return
                fps = self.fps_tracker.update()
                if fps <= 0.0 and getattr(self.camera, "capture_fps", 0) > 0:
                    fps = self.camera.capture_fps
                self.last_frame = frame
                self.frame_ready.emit(frame)
                self.fps_ready.emit(fps)
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
            self.inspection_state.emit("INSPECTING")
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

            # Send hardware result signal to Arduino Due (conveyor + diverter control)
            result_acknowledged = True
            if self.serial_bridge:
                result_acknowledged = self.serial_bridge.send_result(
                    "OK" if result["ok"] else "NG"
                )
                if not result_acknowledged:
                    self.log.emit("Arduino did not acknowledge the inspection result; check controller state and serial connection.")
            # Inspection finished
            self.pcb_stop_sent=False
            self.inspect_waiting=False        
            self._ng_hold = not result["ok"] and result_acknowledged
            self._ng_removed_at = None
            self._ng_absence_votes.clear()
            if not result["ok"] and not result_acknowledged:
                self._stop_fault = True

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
            if result_acknowledged:
                self.inspection_state.emit("PASS" if result["ok"] else "NG_WAIT_REMOVE")
            else:
                self.inspection_state.emit("RESULT_UNCONFIRMED")

        except Exception as e:
            self.inspection_state.emit("ERROR")
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
        if self.serial_bridge:
            self.serial_bridge.send_reset()
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

            if self.serial_bridge:
                self.serial_bridge.disconnect()

            if self.ros_node:
                self.ros_node.close()

            if self.detector:
                self.detector.close()

            self.log.emit("Inference Worker cleanly shut down.")

        except Exception as e:
            self.error.emit(f"Shutdown error: {e}")

