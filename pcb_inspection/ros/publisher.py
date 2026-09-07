import time
import json
import logging
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from .manager import RosManager
from ..config import (
    ROS_NODE_NAME,
    ROS_TOPIC_IMAGE_RAW,
    ROS_TOPIC_IMAGE_ANNOTATED,
    ROS_TOPIC_PCB_RESULT,
    ROS_TOPIC_INSPECTION_RESULT,
    ROS_TOPIC_STATUS,
    ROS_TOPIC_COMMAND,
    ROS_PUBLISH_RAW_IMAGE,
    ROS_PUBLISH_ANNOTATED_IMAGE,
    ROS_PUBLISH_MAX_FPS,
)

logger = logging.getLogger("AoiRosNode")


class AoiRosNode(Node):
    """
    ROS 2 Node for AOI PCB defect inspection system.
    Publishes images and inspection results; receives control commands.
    """

    def __init__(self, node_name: str = ROS_NODE_NAME):
        super().__init__(node_name)

        self.bridge = CvBridge()
        self.command_callback = None

        # Custom QoS for control messages
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Publishers
        self.pub_raw_image = self.create_publisher(
            Image,
            ROS_TOPIC_IMAGE_RAW,
            qos_profile_sensor_data,
        )

        self.pub_annotated_image = self.create_publisher(
            Image,
            ROS_TOPIC_IMAGE_ANNOTATED,
            qos_profile_sensor_data,
        )

        self.pub_pcb_result = self.create_publisher(
            String,
            ROS_TOPIC_PCB_RESULT,
            reliable_qos,
        )

        self.pub_inspection_result = self.create_publisher(
            String,
            ROS_TOPIC_INSPECTION_RESULT,
            reliable_qos,
        )

        self.pub_status = self.create_publisher(
            String,
            ROS_TOPIC_STATUS,
            reliable_qos,
        )

        # Subscriber for external commands
        self.sub_command = self.create_subscription(
            String,
            ROS_TOPIC_COMMAND,
            self._on_command_received,
            reliable_qos,
        )

        # Throttling
        self.publish_raw = ROS_PUBLISH_RAW_IMAGE
        self.publish_annotated = ROS_PUBLISH_ANNOTATED_IMAGE
        self.min_image_interval = 1.0 / max(1.0, ROS_PUBLISH_MAX_FPS)
        self.last_raw_pub_time = 0.0
        self.last_anno_pub_time = 0.0

        # Register to background executor
        RosManager.instance().add_node(self)
        self.get_logger().info(
            f"AOI ROS 2 Node initialized on topic '{ROS_TOPIC_PCB_RESULT}' and '{ROS_TOPIC_COMMAND}'"
        )

    def set_command_callback(self, callback):
        """Set callback invoked when external ROS command is received."""
        self.command_callback = callback

    def _on_command_received(self, msg: String):
        """Handle incoming command from ROS 2."""
        cmd = msg.data.strip().upper()
        self.get_logger().info(f"Received ROS 2 command: '{cmd}'")
        if self.command_callback is not None:
            try:
                self.command_callback(cmd)
            except Exception as e:
                self.get_logger().error(f"Error handling command '{cmd}': {e}")

    def publish_inspection(
        self,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        result: dict,
        fps: float = 0.0,
    ):
        """Publish detection results, OK/NG signal, and throttled images."""
        now = time.perf_counter()

        # 1. Publish /pcb_result ("OK" or "NG")
        ok = result.get("ok", len(result.get("detections", [])) == 0)
        res_str = "OK" if ok else "NG"

        msg_result = String()
        msg_result.data = res_str
        self.pub_pcb_result.publish(msg_result)

        # 2. Publish detailed /aoi/inspection_result JSON
        payload = {
            "status": res_str,
            "ok": ok,
            "defect_count": len(result.get("detections", [])),
            "stats": result.get("stats", {}),
            "detections": result.get("detections", []),
            "inference_ms": round(result.get("inference_ms", 0.0), 2),
            "fps": round(fps, 1),
            "timestamp": time.time(),
        }

        msg_json = String()
        msg_json.data = json.dumps(payload)
        self.pub_inspection_result.publish(msg_json)

        # 3. Publish Raw Image (throttled & only if subscribers exist)
        if self.publish_raw and raw_frame is not None and self.pub_raw_image.get_subscription_count() > 0:
            if now - self.last_raw_pub_time >= self.min_image_interval:
                try:
                    img_msg = self.bridge.cv2_to_imgmsg(raw_frame, encoding="bgr8")
                    self.pub_raw_image.publish(img_msg)
                    self.last_raw_pub_time = now
                except Exception as e:
                    self.get_logger().warn(f"Failed to publish raw image: {e}")

        # 4. Publish Annotated Image (throttled & only if subscribers exist)
        if self.publish_annotated and annotated_frame is not None and self.pub_annotated_image.get_subscription_count() > 0:
            if now - self.last_anno_pub_time >= self.min_image_interval:
                try:
                    anno_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                    self.pub_annotated_image.publish(anno_msg)
                    self.last_anno_pub_time = now
                except Exception as e:
                    self.get_logger().warn(f"Failed to publish annotated image: {e}")

    def publish_status(self, status_dict: dict):
        """Publish system heartbeat status."""
        msg = String()
        msg.data = json.dumps(status_dict)
        self.pub_status.publish(msg)

    def close(self):
        """Cleanly remove node from manager."""
        try:
            RosManager.instance().remove_node(self)
        except Exception:
            pass


# Backward compatibility alias
ImagePublisher = AoiRosNode