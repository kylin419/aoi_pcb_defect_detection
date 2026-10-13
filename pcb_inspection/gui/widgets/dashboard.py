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

        root.addWidget(self._inspection_status_group())
        root.addWidget(self._production_group())
        root.addWidget(self._performance_group())
        root.addWidget(self._detection_group())
        root.addWidget(self._system_group())

        root.addStretch()

    def _inspection_status_group(self):
        box = QGroupBox("Inspection Status")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        self.inspection_status = QLabel("待機中")
        self.inspection_status.setObjectName("inspectionStatus")
        self.inspection_status.setStyleSheet("font-size: 16pt; font-weight: bold; color: #00D084;")
        self.inspection_detail = QLabel("按下開始後，輸送帶運轉並由鏡頭搜尋 PCB。")
        self.inspection_detail.setObjectName("inspectionDetail")
        self.inspection_detail.setWordWrap(True)
        layout.addWidget(self.inspection_status)
        layout.addWidget(self.inspection_detail)
        return box

    def set_inspection_state(self, state: str):
        states = {
            "IDLE": ("待機中", "按下開始後，輸送帶運轉並由鏡頭搜尋 PCB。", "#00D084"),
            "SEARCHING": ("搜尋 PCB 中", "輸送帶運轉中，鏡頭正在確認 PCB 是否進入畫面。", "#00D2FF"),
            "PCB_DETECTED": ("已偵測到 PCB", "正在通知 Arduino 停止輸送帶。", "#FFB300"),
            "STOP_CONFIRMED": ("輸送帶已停止", "Arduino 已完成置中運轉並確認停帶，準備開始辨識。", "#00D084"),
            "CENTERING": ("PCB 置中運轉中", "輸送帶以較慢速度移動 PCB；到達設定時間後停帶並開始辨識。", "#FFB300"),
            "INSPECTING": ("瑕疵辨識中", "正在分析目前影格，請稍候。", "#FFB300"),
            "PASS": ("辨識完成：良品 OK", "瑕疵檢查通過，Arduino 已確認收到結果。", "#00D084"),
            "FAIL": ("辨識完成：瑕疵 NG", "檢查到瑕疵，Arduino 已確認收到結果。", "#FF5555"),
            "NG_WAIT_REMOVE": ("瑕疵板待移除", "請移除瑕疵 PCB；鏡頭確認畫面清空後，等待約 2 秒恢復輸送帶。", "#FF5555"),
            "NG_RESUME_DELAY": ("等待恢復輸送帶", "已確認瑕疵 PCB 離開畫面，約 2 秒後自動恢復。", "#FFB300"),
            "RESUME_UNCONFIRMED": ("恢復運轉未確認", "Arduino 未確認恢復輸送帶；設備維持故障狀態，請檢查控制器。", "#FF5555"),
            "STOP_UNCONFIRMED": ("停帶未確認", "未收到 Arduino 停帶確認；請檢查控制器連線與馬達狀態。", "#FF5555"),
            "RESULT_UNCONFIRMED": ("辨識結果未確認", "未收到 Arduino 結果確認；請檢查控制器狀態與序列連線。", "#FF5555"),
            "ERROR": ("辨識發生錯誤", "請查看下方系統記錄並確認相機與模型狀態。", "#FF5555"),
        }
        title, detail, color = states.get(state, (state, "", "#CCCCCC"))
        self.inspection_status.setText(title)
        self.inspection_status.setStyleSheet(
            f"font-size: 16pt; font-weight: bold; color: {color};"
        )
        self.inspection_detail.setText(detail)

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
