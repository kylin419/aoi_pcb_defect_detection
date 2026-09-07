from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QSlider,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QPushButton,
    QDialogButtonBox,
)

from ...config import (
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    CAMERA_SOURCE,
    AUTO_SAVE_NG,
    ROS_TOPIC_PCB_RESULT,
    ROS_TOPIC_INSPECTION_RESULT,
    ROS_TOPIC_COMMAND,
)


class SettingsDialog(QDialog):
    """Configuration dialog for detection thresholds, camera, and ROS 2 options."""

    settings_applied = pyqtSignal(dict)

    def __init__(
        self,
        parent=None,
        conf_thresh: float = CONFIDENCE_THRESHOLD,
        iou_thresh: float = IOU_THRESHOLD,
        cam_source: str = CAMERA_SOURCE,
        auto_save_ng: bool = AUTO_SAVE_NG,
    ):
        super().__init__(parent)

        self.setWindowTitle("AOI System Settings")
        self.resize(460, 480)

        self.current_conf = conf_thresh
        self.current_iou = iou_thresh
        self.current_cam = cam_source
        self.current_auto_save = auto_save_ng

        self.build_ui()
        self.load_values()

    def build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)

        # 1. Detection Group
        det_group = QGroupBox("YOLOv12 Defect Detection")
        det_layout = QVBoxLayout(det_group)
        det_layout.setSpacing(10)

        # Confidence
        conf_box = QHBoxLayout()
        conf_label = QLabel("Confidence Threshold:")
        conf_label.setFixedWidth(160)
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setFixedWidth(70)

        conf_box.addWidget(conf_label)
        conf_box.addWidget(self.conf_slider)
        conf_box.addWidget(self.conf_spin)
        det_layout.addLayout(conf_box)

        # IoU
        iou_box = QHBoxLayout()
        iou_label = QLabel("NMS IoU Threshold:")
        iou_label.setFixedWidth(160)
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setRange(10, 95)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.10, 0.95)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setFixedWidth(70)

        iou_box.addWidget(iou_label)
        iou_box.addWidget(self.iou_slider)
        iou_box.addWidget(self.iou_spin)
        det_layout.addLayout(iou_box)

        # Auto Save NG
        self.auto_save_box = QCheckBox("Automatically save defect images and JSON on NG")
        det_layout.addWidget(self.auto_save_box)

        root.addWidget(det_group)

        # 2. Camera Group
        cam_group = QGroupBox("Camera Source")
        cam_layout = QHBoxLayout(cam_group)

        cam_label = QLabel("Active Video Source:")
        cam_label.setFixedWidth(160)
        self.cam_combo = QComboBox()
        self.cam_combo.addItem("Auto (CSI -> USB -> Simulation)", "auto")
        self.cam_combo.addItem("Jetson CSI Camera (IMX477)", "csi")
        self.cam_combo.addItem("USB Webcam (/dev/video0)", "usb")
        self.cam_combo.addItem("Simulation Mode (test.jpg)", "simulation")

        cam_layout.addWidget(cam_label)
        cam_layout.addWidget(self.cam_combo)
        root.addWidget(cam_group)

        # 3. ROS 2 Info Group
        ros_group = QGroupBox("ROS 2 Integration")
        ros_layout = QVBoxLayout(ros_group)
        ros_layout.setSpacing(6)

        ros_layout.addWidget(QLabel(f"<b>Result Topic:</b> <font color='#00ff7f'>{ROS_TOPIC_PCB_RESULT}</font> (std_msgs/String)"))
        ros_layout.addWidget(QLabel(f"<b>JSON Result:</b> <font color='#00ff7f'>{ROS_TOPIC_INSPECTION_RESULT}</font> (JSON payload)"))
        ros_layout.addWidget(QLabel(f"<b>Command Topic:</b> <font color='#00d2ff'>{ROS_TOPIC_COMMAND}</font> (START, STOP, TRIGGER, etc.)"))
        root.addWidget(ros_group)

        # 4. Buttons
        btn_box = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")

        self.apply_btn.clicked.connect(self.on_apply)
        self.ok_btn.clicked.connect(self.on_ok)
        self.cancel_btn.clicked.connect(self.reject)

        btn_box.addStretch()
        btn_box.addWidget(self.apply_btn)
        btn_box.addWidget(self.ok_btn)
        btn_box.addWidget(self.cancel_btn)
        root.addLayout(btn_box)

        # Connect Sliders & Spins
        self.conf_slider.valueChanged.connect(lambda v: self.conf_spin.setValue(v / 100.0))
        self.conf_spin.valueChanged.connect(lambda v: self.conf_slider.setValue(int(v * 100)))

        self.iou_slider.valueChanged.connect(lambda v: self.iou_spin.setValue(v / 100.0))
        self.iou_spin.valueChanged.connect(lambda v: self.iou_slider.setValue(int(v * 100)))

    def load_values(self):
        self.conf_spin.setValue(self.current_conf)
        self.iou_spin.setValue(self.current_iou)
        self.auto_save_box.setChecked(self.current_auto_save)

        # Set combo
        idx = self.cam_combo.findData(self.current_cam)
        if idx >= 0:
            self.cam_combo.setCurrentIndex(idx)

    def get_data(self) -> dict:
        return {
            "confidence": self.conf_spin.value(),
            "iou": self.iou_spin.value(),
            "camera_source": self.cam_combo.currentData(),
            "auto_save_ng": self.auto_save_box.isChecked(),
        }

    def on_apply(self):
        data = self.get_data()
        self.settings_applied.emit(data)

    def on_ok(self):
        self.on_apply()
        self.accept()
