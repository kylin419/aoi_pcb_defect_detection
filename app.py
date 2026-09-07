import os
import sys

# Ensure Jetson static TLS allocation fix is active for GStreamer & TensorRT
if os.path.exists("/lib/aarch64-linux-gnu/libGLdispatch.so.0"):
    cur_preload = os.environ.get("LD_PRELOAD", "")
    if "/lib/aarch64-linux-gnu/libGLdispatch.so.0" not in cur_preload:
        os.environ["LD_PRELOAD"] = f"/lib/aarch64-linux-gnu/libGLdispatch.so.0:{cur_preload}".strip(":")
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            pass

import rclpy
from PyQt6.QtWidgets import QApplication

from pcb_inspection.gui.window import MainWindow
from pcb_inspection.gui.theme import load_theme


def main():
    # Initialize ROS 2
    if not rclpy.ok():
        rclpy.init(args=sys.argv)

    # Initialize Qt Application
    app = QApplication(sys.argv)
    load_theme(app)

    window = MainWindow()
    window.show()

    # Enter Qt main event loop
    exit_code = app.exec()

    # Clean shutdown of ROS 2
    if rclpy.ok():
        rclpy.shutdown()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()