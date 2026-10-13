from PyQt6.QtWidgets import (
    QStatusBar,
    QLabel,
    QWidget,
    QHBoxLayout,
)


class StatusItem(QWidget):

    def __init__(self, name):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(4)

        self.name = QLabel(f"{name}:")
        self.value = QLabel("--")

        self.name.setObjectName("statusName")
        self.value.setObjectName("statusValue")

        layout.addWidget(self.name)
        layout.addWidget(self.value)

    def set_value(self, value):
        self.value.setText(str(value))

    def set_state(self, text, color):
        self.value.setText(f"<font color='{color}'>●</font> {text}")


class StatusBarWidget(QStatusBar):

    def __init__(self):
        super().__init__()

        self.ros = StatusItem("ROS 2")
        self.camera = StatusItem("Camera")
        self.engine = StatusItem("TensorRT")
        self.mode = StatusItem("Mode")
        self.inspection = StatusItem("Inspection")

        self.addPermanentWidget(self.inspection)
        self.addPermanentWidget(self.ros)
        self.addPermanentWidget(self.camera)
        self.addPermanentWidget(self.engine)
        self.addPermanentWidget(self.mode)

        self.set_ros(True, "Active")
        self.set_camera("Checking...")
        self.set_engine(False)
        self.set_mode("Standby")
        self.set_inspection_status("IDLE")

    def set_inspection_status(self, state: str):
        labels = {
            "IDLE": ("待機", "#00D084"),
            "SEARCHING": ("搜尋 PCB", "#00D2FF"),
            "PCB_DETECTED": ("已偵測 PCB", "#FFB300"),
            "STOP_CONFIRMED": ("已停帶", "#00D084"),
            "CENTERING": ("PCB置中運轉", "#FFB300"),
            "INSPECTING": ("辨識中", "#FFB300"),
            "PASS": ("良品 OK", "#00D084"),
            "FAIL": ("瑕疵 NG", "#FF5555"),
            "NG_WAIT_REMOVE": ("待移除瑕疵板", "#FF5555"),
            "NG_RESUME_DELAY": ("2秒後恢復", "#FFB300"),
            "RESUME_UNCONFIRMED": ("恢復未確認", "#FF5555"),
            "STOP_UNCONFIRMED": ("停帶未確認", "#FF5555"),
            "RESULT_UNCONFIRMED": ("結果未確認", "#FF5555"),
            "ERROR": ("辨識錯誤", "#FF5555"),
        }
        text, color = labels.get(state, (state, "#CCCCCC"))
        self.inspection.set_state(text, color)

    def set_ros(self, active: bool, text: str = "Active"):
        if active:
            self.ros.set_state(text, "#00ff7f")
        else:
            self.ros.set_state("Offline", "#ff5555")

    def set_camera(self, mode: str):
        if "Disconnected" in mode or "Error" in mode:
            self.camera.set_state(mode, "#ff5555")
        elif "Simulation" in mode:
            self.camera.set_state(mode, "#00d2ff")
        else:
            self.camera.set_state(mode, "#00ff7f")

    def set_engine(self, loaded: bool):
        if loaded:
            self.engine.set_state("Ready", "#00ff7f")
        else:
            self.engine.set_state("Not Loaded", "#ffaa00")

    def set_mode(self, text: str):
        if text == "Continuous":
            self.mode.set_state("Continuous", "#00ff7f")
        elif text == "Trigger":
            self.mode.set_state("Single-Trigger", "#ffaa00")
        else:
            self.mode.set_state("Live Preview", "#00d2ff")
