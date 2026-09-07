import cv2
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import QWidget


class VideoWidget(QWidget):

    def __init__(self):
        super().__init__()

        self.setMinimumSize(960, 640)

        self._frame = None
        self._pixmap = None
        self._fps = 0.0
        self._res_str = "--"

    # ---------------------------
    # Public API
    # ---------------------------

    def set_frame(self, frame: np.ndarray, fps: float = None):
        """Update latest OpenCV frame and pre-convert for smooth rendering."""
        if frame is None:
            self._frame = None
            self._pixmap = None
            self.update()
            return

        self._frame = frame
        h, w = frame.shape[:2]
        self._res_str = f"{w} x {h}"

        if fps is not None:
            self._fps = max(0.0, float(fps))

        # Fast conversion from BGR to QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(
            rgb.data,
            w,
            h,
            3 * w,
            QImage.Format.Format_RGB888,
        )
        self._pixmap = QPixmap.fromImage(image)
        self.update()

    def set_fps(self, fps: float):
        """Update display FPS."""
        try:
            val = float(fps)
            if val != self._fps:
                self._fps = max(0.0, val)
                self.update()
        except (ValueError, TypeError):
            pass

    # ---------------------------
    # Paint
    # ---------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#181818"))

        if self._pixmap is None or self._pixmap.isNull():
            self._draw_placeholder(painter)
            return

        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )

        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2

        painter.drawPixmap(x, y, scaled)

        self._draw_overlay(
            painter,
            x,
            y,
            scaled.width(),
            scaled.height(),
        )

    # ---------------------------
    # Helpers
    # ---------------------------

    def _draw_placeholder(self, painter):
        painter.setPen(QColor("#AAAAAA"))
        font = QFont()
        font.setPointSize(18)
        painter.setFont(font)
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            "Waiting for Camera Stream...",
        )

    def _draw_overlay(
        self,
        painter,
        x,
        y,
        w,
        h,
    ):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 180))

        painter.drawRoundedRect(
            x + 12,
            y + 12,
            190,
            60,
            6,
            6,
        )

        fps_color = "#00FF7F" if self._fps >= 15.0 else ("#FFB300" if self._fps > 0.0 else "#FF5555")
        painter.setPen(QPen(QColor(fps_color)))

        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)

        painter.drawText(
            x + 22,
            y + 35,
            f"FPS : {self._fps:.1f}",
        )

        font.setBold(False)
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#CCCCCC")))

        painter.drawText(
            x + 22,
            y + 56,
            f"Resolution : {self._res_str}",
        )