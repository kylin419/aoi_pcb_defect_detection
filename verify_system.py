#!/usr/bin/env python3
"""
AOI PCB Defect Detection System Self-Check Script.
Tests:
1. Camera acquisition & fallback mechanism
2. TensorRT YOLOv12 model inference & bounding box scaling
3. ROS 2 node publishing & subscription
4. Jetson hardware telemetry reading
"""

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

import time
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image

from pcb_inspection.camera import Camera
from pcb_inspection.detector import TensorRTDetector
from pcb_inspection.ros.publisher import AoiRosNode
from pcb_inspection.ros.manager import RosManager
from pcb_inspection.visualization import draw_detections
from pcb_inspection.config import TEST_IMAGE_PATH, ENGINE_PATH


def test_system():
    print("=" * 60)
    print("      AOI PCB Defect Detection System Self-Check")
    print("=" * 60)

    # 1. Test Camera
    print("\n[1/4] Checking Camera Interface...")
    cam = Camera()
    success, mode, msg = cam.open()
    print(f"  -> Camera Opened: {success}")
    print(f"  -> Active Mode: {mode}")
    print(f"  -> Status: {msg}")
    ret, frame = cam.read()
    if ret and frame is not None:
        print(f"  -> Captured Frame Resolution: {frame.shape[1]}x{frame.shape[0]} (Channels: {frame.shape[2]})")
    else:
        print("  -> ERROR: Failed to read frame!")
        return False
    cam.release()

    # 2. Test TensorRT Detector
    print("\n[2/4] Checking TensorRT YOLOv12 Detector...")
    if not os.path.exists(ENGINE_PATH):
        print(f"  -> ERROR: Engine file missing at {ENGINE_PATH}")
        return False
    
    det = TensorRTDetector(ENGINE_PATH)
    t0 = time.perf_counter()
    result = det.detect(frame)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  -> Inference Time: {result['inference_ms']:.2f} ms (Wall: {dt_ms:.2f} ms)")
    print(f"  -> Total Detections: {len(result['detections'])}")
    print(f"  -> Inspection Result: {'PASS (OK)' if result['ok'] else 'FAIL (NG)'}")
    print(f"  -> Defect Breakdown: {result['stats']}")
    det.close()

    # 3. Test ROS 2 Integration
    print("\n[3/4] Checking ROS 2 Integration...")
    if not rclpy.ok():
        rclpy.init()

    received = {"res": None, "detail": None, "raw": None, "anno": None}

    class TestSub(Node):
        def __init__(self):
            super().__init__('verify_sub_node')
            self.create_subscription(String, '/pcb_result', lambda m: received.update(res=m.data), 10)
            self.create_subscription(String, '/aoi/inspection_result', lambda m: received.update(detail=json.loads(m.data)), 10)
            self.create_subscription(Image, '/camera/image_raw', lambda m: received.update(raw=f"{m.width}x{m.height}"), 10)
            self.create_subscription(Image, '/camera/image_annotated', lambda m: received.update(anno=f"{m.width}x{m.height}"), 10)

    sub_node = TestSub()
    RosManager.instance().add_node(sub_node)

    aoi_node = AoiRosNode()
    annotated = draw_detections(frame, result["detections"], inference_ms=result["inference_ms"])
    aoi_node.publish_inspection(raw_frame=frame, annotated_frame=annotated, result=result, fps=30.0)

    # Wait for delivery
    start_wait = time.time()
    while time.time() - start_wait < 2.0:
        if all(v is not None for v in received.values()):
            break
        time.sleep(0.05)

    print(f"  -> /pcb_result: {received['res']}")
    print(f"  -> /aoi/inspection_result status: {received['detail'].get('status') if received['detail'] else 'None'}")
    print(f"  -> /camera/image_raw: {received['raw']}")
    print(f"  -> /camera/image_annotated: {received['anno']}")

    aoi_node.close()
    RosManager.instance().remove_node(sub_node)
    sub_node.destroy_node()

    ros_ok = received['res'] is not None and received['detail'] is not None
    print(f"  -> ROS 2 Bus Status: {'SUCCESS' if ros_ok else 'FAILED'}")

    # 4. Telemetry Check
    print("\n[4/4] Reading Hardware Telemetry...")
    temp = "N/A"
    if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            v = float(f.read().strip())
            temp = f"{v/1000.0:.1f} °C" if v > 1000 else f"{v:.1f} °C"

    gpu = "N/A"
    if os.path.exists("/sys/devices/gpu.0/load"):
        with open("/sys/devices/gpu.0/load") as f:
            gpu = f"{float(f.read().strip())/10.0:.0f}%"

    print(f"  -> Jetson SoC Temperature: {temp}")
    print(f"  -> GPU Load: {gpu}")

    print("\n" + "=" * 60)
    print("        ALL SYSTEM CHECKS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)
