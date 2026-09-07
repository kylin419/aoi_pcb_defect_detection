from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QPushButton,
)


class ToolBarWidget(QWidget):
    """Top toolbar for AOI PCB Inspection."""

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    trigger_clicked = pyqtSignal()
    snapshot_clicked = pyqtSignal()
    save_clicked = pyqtSignal()
    reset_stats_clicked = pyqtSignal()
    reload_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()
    exit_clicked = pyqtSignal()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()

        self.build_ui()
        self.connect_signals()
        self.set_running(False)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.start_btn = self.create_button(
            "▶ Start",
            "Start Continuous Inspection",
            object_name="primary",
        )

        self.stop_btn = self.create_button(
            "■ Stop",
            "Stop Inspection",
        )

        self.trigger_btn = self.create_button(
            "⚡ Trigger",
            "Single-Shot Inspection (Trigger Mode)",
        )

        self.snapshot_btn = self.create_button(
            "📷 Snapshot",
            "Save Current Frame",
        )

        self.save_btn = self.create_button(
            "💾 Save",
            "Save Detection Result",
        )

        self.reset_stats_btn = self.create_button(
            "📊 Reset",
            "Reset Yield Statistics",
        )

        self.reload_btn = self.create_button(
            "🔄 Reload",
            "Reload TensorRT Engine",
        )

        self.settings_btn = self.create_button(
            "⚙ Settings",
            "Application Settings",
        )

        self.exit_btn = self.create_button(
            "✕ Exit",
            "Exit Application",
        )

        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.trigger_btn)

        layout.addSpacing(8)

        layout.addWidget(self.snapshot_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.reset_stats_btn)

        layout.addStretch()

        layout.addWidget(self.reload_btn)
        layout.addWidget(self.settings_btn)

        layout.addSpacing(8)

        layout.addWidget(self.exit_btn)

    # ------------------------------------------------------------------
    # Button Factory
    # ------------------------------------------------------------------

    def create_button(
        self,
        text: str,
        tooltip: str,
        object_name: str = "",
    ) -> QPushButton:
        button = QPushButton(text)
        if object_name:
            button.setObjectName(object_name)
        button.setToolTip(tooltip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setMinimumHeight(38)
        button.setMinimumWidth(100)
        return button

    # ------------------------------------------------------------------
    # Signal Connections
    # ------------------------------------------------------------------

    def connect_signals(self):
        self.start_btn.clicked.connect(self.start_clicked.emit)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        self.trigger_btn.clicked.connect(self.trigger_clicked.emit)
        self.snapshot_btn.clicked.connect(self.snapshot_clicked.emit)
        self.save_btn.clicked.connect(self.save_clicked.emit)
        self.reset_stats_btn.clicked.connect(self.reset_stats_clicked.emit)
        self.reload_btn.clicked.connect(self.reload_clicked.emit)
        self.settings_btn.clicked.connect(self.settings_clicked.emit)
        self.exit_btn.clicked.connect(self.exit_clicked.emit)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def set_running(self, running: bool):
        """Update toolbar buttons according to inspection state."""
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.trigger_btn.setEnabled(not running)

        self.snapshot_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.reset_stats_btn.setEnabled(True)

        self.reload_btn.setEnabled(not running)
        self.settings_btn.setEnabled(not running)