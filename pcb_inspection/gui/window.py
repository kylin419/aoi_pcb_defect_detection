from PyQt6.QtCore import Qt, QThread, QMetaObject
from PyQt6.QtWidgets import (
    QWidget,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
)

from .widgets.video_widget import VideoWidget
from .widgets.dashboard import Dashboard
from .widgets.toolbar import ToolBarWidget
from .widgets.log_widget import LogWidget
from .widgets.statusbar import StatusBarWidget
from .dialogs.settings_dialog import SettingsDialog

from ..workers.inference_worker import InferenceWorker
from ..ros.manager import RosManager
from ..config import (
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    CAMERA_SOURCE,
    AUTO_SAVE_NG,
)


class MainWindow(QMainWindow):
    """Main AOI PCB Inspection application window."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AOI PCB Inspection System [YOLOv12 & ROS 2]")
        self.resize(1600, 920)

        self.thread: QThread | None = None
        self.worker: InferenceWorker | None = None

        # Settings state
        self.conf_threshold = CONFIDENCE_THRESHOLD
        self.iou_threshold = IOU_THRESHOLD
        self.camera_source = CAMERA_SOURCE
        self.auto_save_ng = AUTO_SAVE_NG

        self.build_ui()
        self.build_worker()
        self.connect_signals()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Top area: Video viewer (left) + Inspection Dashboard (right)
        top = QHBoxLayout()
        self.video = VideoWidget()
        self.dashboard = Dashboard()

        top.addWidget(self.video, 3)
        top.addWidget(self.dashboard, 1)
        root.addLayout(top, 4)

        # Bottom area: Action toolbar + Log console
        self.toolbar = ToolBarWidget()
        root.addWidget(self.toolbar)

        self.log = LogWidget()
        root.addWidget(self.log, 1)

        # Status bar
        self.status = StatusBarWidget()
        self.setStatusBar(self.status)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def build_worker(self):
        self.thread = QThread(self)
        self.worker = InferenceWorker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.initialize)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def connect_signals(self):
        # Toolbar Actions
        self.toolbar.start_clicked.connect(self.worker.start)
        self.toolbar.stop_clicked.connect(self.worker.stop)
        self.toolbar.trigger_clicked.connect(self.worker.trigger)
        self.toolbar.snapshot_clicked.connect(self.worker.snapshot)
        self.toolbar.save_clicked.connect(self.worker.save_result)
        self.toolbar.reset_stats_clicked.connect(self.worker.reset_statistics)
        self.toolbar.reload_clicked.connect(self.on_reload_model)
        self.toolbar.settings_clicked.connect(self.on_settings)
        self.toolbar.exit_clicked.connect(self.close)

        # Worker to UI
        self.worker.frame_ready.connect(self.video.set_frame)
        self.worker.fps_ready.connect(self.video.set_fps)
        self.worker.dashboard_ready.connect(self.on_dashboard_update)
        self.worker.status_ready.connect(self.on_status_ready)

        self.worker.log.connect(self.log.info)
        self.worker.error.connect(self.on_error)
        self.worker.started.connect(self.on_started)
        self.worker.stopped.connect(self.on_stopped)
        self.worker.ros_command_received.connect(self.on_ros_command)

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def on_dashboard_update(self, data: dict):
        self.dashboard.update(data)
        if "fps" in data:
            try:
                self.video.set_fps(float(data["fps"]))
            except (ValueError, TypeError):
                pass

    def on_started(self):
        self.toolbar.set_running(True)
        self.status.set_engine(True)
        self.status.set_mode("Continuous")
        self.log.success("Continuous Inspection Running.")

    def on_stopped(self):
        self.toolbar.set_running(False)
        self.status.set_mode("Standby")
        self.log.warning("Inspection Paused (Live Camera Streaming).")

    def on_status_ready(self, status: dict):
        if "camera_mode" in status:
            self.status.set_camera(status["camera_mode"])
        if "engine" in status:
            self.status.set_engine(status["engine"])
        if "ros" in status:
            self.status.set_ros(status["ros"])

    def on_ros_command(self, cmd: str):
        self.log.info(f"<b>[ROS 2 Command]</b> Executed: <code>{cmd}</code>")

    def on_error(self, message: str):
        self.log.error(message)

    # ------------------------------------------------------------------
    # User Actions
    # ------------------------------------------------------------------

    def on_reload_model(self):
        self.log.info("Requesting TensorRT engine reload...")
        QMetaObject.invokeMethod(
            self.worker,
            "reload_detector",
            Qt.ConnectionType.QueuedConnection,
        )

    def on_settings(self):
        dlg = SettingsDialog(
            parent=self,
            conf_thresh=self.conf_threshold,
            iou_thresh=self.iou_threshold,
            cam_source=self.camera_source,
            auto_save_ng=self.auto_save_ng,
        )
        dlg.settings_applied.connect(self.apply_settings)
        dlg.exec()

    def apply_settings(self, data: dict):
        self.conf_threshold = data.get("confidence", self.conf_threshold)
        self.iou_threshold = data.get("iou", self.iou_threshold)
        new_cam = data.get("camera_source", self.camera_source)
        self.auto_save_ng = data.get("auto_save_ng", self.auto_save_ng)

        # Update worker
        if self.worker is not None:
            self.worker.set_thresholds(self.conf_threshold, self.iou_threshold)
            self.worker.set_auto_save_ng(self.auto_save_ng)
            if new_cam != self.camera_source:
                self.camera_source = new_cam
                self.worker.set_camera_source(new_cam)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self.isVisible() and not getattr(self, "_force_close", False):
            reply = QMessageBox.question(
                self,
                "Exit AOI System",
                "Are you sure you want to stop inspection and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return


        if self.worker is not None:
            QMetaObject.invokeMethod(
                self.worker,
                "stop",
                Qt.ConnectionType.BlockingQueuedConnection,
            )
            QMetaObject.invokeMethod(
                self.worker,
                "shutdown",
                Qt.ConnectionType.BlockingQueuedConnection,
            )

        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(2000)

        # Cleanly stop ROS 2 background manager
        try:
            RosManager.instance().shutdown()
        except Exception:
            pass

        event.accept()

