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

        self.addPermanentWidget(self.ros)
        self.addPermanentWidget(self.camera)
        self.addPermanentWidget(self.engine)
        self.addPermanentWidget(self.mode)

        self.set_ros(True, "Active")
        self.set_camera("Checking...")
        self.set_engine(False)
        self.set_mode("Standby")

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