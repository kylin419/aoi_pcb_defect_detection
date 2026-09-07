import threading
import logging
import rclpy
from rclpy.executors import MultiThreadedExecutor

logger = logging.getLogger("RosManager")


class RosManager:
    """Manages ROS 2 background executor thread and node registration."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        if not rclpy.ok():
            rclpy.init()

        self.executor = MultiThreadedExecutor(num_threads=2)
        self.running = True
        self.nodes = []

        self.thread = threading.Thread(
            target=self._spin,
            name="ROS2-Executor-Thread",
            daemon=True,
        )
        self.thread.start()

    def _spin(self):
        while self.running and rclpy.ok():
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                if self.running:
                    logger.error(f"ROS 2 spin error: {e}")
                break

    @classmethod
    def instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = RosManager()
            return cls._instance

    def add_node(self, node):
        with self._lock:
            if node not in self.nodes:
                self.executor.add_node(node)
                self.nodes.append(node)

    def remove_node(self, node):
        with self._lock:
            if node in self.nodes:
                self.executor.remove_node(node)
                self.nodes.remove(node)

    def shutdown(self):
        with self._lock:
            self.running = False

            for node in self.nodes:
                try:
                    self.executor.remove_node(node)
                    node.destroy_node()
                except Exception:
                    pass
            self.nodes.clear()

            try:
                self.executor.shutdown()
            except Exception:
                pass

            if self.thread.is_alive():
                self.thread.join(timeout=1.0)

            RosManager._instance = None