from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
)


class InfoCard(QWidget):
    """
    Dashboard Information Card

    --------------------------
    FPS                58.4
    --------------------------
    """

    def __init__(self, title: str, value: str = "--"):
        super().__init__()

        self.title = QLabel(title)
        self.value = QLabel(value)

        self.title.setObjectName("cardTitle")
        self.value.setObjectName("cardValue")

        layout = QHBoxLayout(self)

        layout.setContentsMargins(8, 6, 8, 6)

        layout.addWidget(self.title)
        layout.addStretch()
        layout.addWidget(self.value)

    def set_value(self, value):

        self.value.setText(str(value))


class Dashboard(QWidget):

    def __init__(self):
        super().__init__()

        self.setMinimumWidth(340)
        self.cards = {}

        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(4, 4, 4, 4)

        root.addWidget(self._production_group())
        root.addWidget(self._performance_group())
        root.addWidget(self._detection_group())
        root.addWidget(self._system_group())

        root.addStretch()

    # =====================================================
    # Production Yield
    # =====================================================

    def _production_group(self):
        box = QGroupBox("Production Yield")
        layout = QVBoxLayout(box)

        self.cards["total_count"] = InfoCard("Total Tested", "0")
        self.cards["ok_count"] = InfoCard("Passed (OK)", "0")
        self.cards["ng_count"] = InfoCard("Defective (NG)", "0")
        self.cards["yield_rate"] = InfoCard("Yield Rate", "100.0%")

        for key in ("total_count", "ok_count", "ng_count", "yield_rate"):
            layout.addWidget(self.cards[key])

        return box

    # =====================================================
    # Performance
    # =====================================================

    def _performance_group(self):
        box = QGroupBox("Performance")
        layout = QVBoxLayout(box)

        self.cards["fps"] = InfoCard("FPS")
        self.cards["inference"] = InfoCard("Inference")
        self.cards["camera"] = InfoCard("Camera")
        self.cards["model"] = InfoCard("Model")

        for key in ("fps", "inference", "camera", "model"):
            layout.addWidget(self.cards[key])

        return box

    # =====================================================
    # Detection
    # =====================================================

    def _detection_group(self):
        box = QGroupBox("Defect Classification")
        layout = QVBoxLayout(box)

        defects = [
            ("missing_hole", "Missing Hole"),
            ("mouse_bite", "Mouse Bite"),
            ("open_circuit", "Open Circuit"),
            ("short", "Short"),
            ("spur", "Spur"),
            ("spurious_copper", "Spurious Copper"),
        ]

        for key, name in defects:
            self.cards[key] = InfoCard(name, "0")
            layout.addWidget(self.cards[key])

        return box

    # =====================================================
    # System
    # =====================================================

    def _system_group(self):
        box = QGroupBox("Jetson System Telemetry")
        layout = QVBoxLayout(box)

        self.cards["gpu"] = InfoCard("GPU Load")
        self.cards["temperature"] = InfoCard("SoC Temp")
        self.cards["memory"] = InfoCard("RAM Memory")

        layout.addWidget(self.cards["gpu"])
        layout.addWidget(self.cards["temperature"])
        layout.addWidget(self.cards["memory"])

        return box

    # =====================================================
    # Update
    # =====================================================

    def set_value(self, key: str, value):
        if key in self.cards:
            self.cards[key].set_value(value)

    def update(self, result: dict):
        # Support alias
        if "missing" in result and "missing_hole" not in result:
            result["missing_hole"] = result["missing"]

        for key, value in result.items():
            if key in self.cards:
                self.cards[key].set_value(value)

    # =====================================================
    # Reset
    # =====================================================

    def reset(self):
        defaults = {
            "total_count": "0",
            "ok_count": "0",
            "ng_count": "0",
            "yield_rate": "100.0%",
            "fps": "--",
            "inference": "--",
            "camera": "Disconnected",
            "model": "--",
            "gpu": "--",
            "temperature": "--",
            "memory": "--",
            "missing_hole": "0",
            "mouse_bite": "0",
            "open_circuit": "0",
            "short": "0",
            "spur": "0",
            "spurious_copper": "0",
        }
        self.update(defaults)