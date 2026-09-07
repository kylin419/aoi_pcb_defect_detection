from datetime import datetime

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QPlainTextEdit


class LogWidget(QPlainTextEdit):

    INFO = "#7FDBFF"
    SUCCESS = "#2ECC71"
    WARNING = "#F1C40F"
    ERROR = "#E74C3C"

    def __init__(self):

        super().__init__()

        self.setReadOnly(True)

        self.setMaximumHeight(180)

        self.document().setMaximumBlockCount(500)

    def _append(self, level, message, color):

        now = datetime.now().strftime("%H:%M:%S")

        html = (
            f"<span style='color:gray'>[{now}]</span> "
            f"<span style='color:{color}'>[{level}]</span> "
            f"{message}"
        )

        self.appendHtml(html)

    def info(self, message):

        self._append(
            "INFO",
            message,
            self.INFO,
        )

    def success(self, message):

        self._append(
            "SUCCESS",
            message,
            self.SUCCESS,
        )

    def warning(self, message):

        self._append(
            "WARNING",
            message,
            self.WARNING,
        )

    def error(self, message):

        self._append(
            "ERROR",
            message,
            self.ERROR,
        )

    def clear_log(self):

        self.clear()