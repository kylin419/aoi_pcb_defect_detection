import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
ENGINE_PATH = os.path.join(BASE_DIR, "best.engine")
ONNX_PATH = os.path.join(BASE_DIR, "best.onnx")
PT_PATH = os.path.join(BASE_DIR, "best.pt")
TEST_IMAGE_PATH = os.path.join(BASE_DIR, "test.jpg")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Defect classes
CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

# Human-readable labels for UI display
CLASS_DISPLAY_NAMES = {
    "missing_hole": "Missing Hole",
    "mouse_bite": "Mouse Bite",
    "open_circuit": "Open Circuit",
    "short": "Short",
    "spur": "Spur",
    "spurious_copper": "Spurious Copper",
}

# Distinct high-contrast BGR colors for visualization
CLASS_COLORS = {
    "missing_hole": (0, 0, 255),       # Red
    "mouse_bite": (0, 140, 255),       # Orange
    "open_circuit": (0, 215, 255),     # Yellow-Gold
    "short": (255, 0, 180),           # Magenta
    "spur": (255, 200, 0),            # Cyan
    "spurious_copper": (200, 100, 255) # Light Pink/Violet
}

# Camera configurations
CAMERA_SOURCE = "auto"  # 'auto', 'csi', 'usb', 'simulation'
CAMERA_SENSOR_ID = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
FRAMERATE = 30

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45
AUTO_SAVE_NG = False

# ROS 2 Topics and Configuration
ROS_NODE_NAME = "aoi_pcb_node"
ROS_TOPIC_IMAGE_RAW = "/camera/image_raw"
ROS_TOPIC_IMAGE_ANNOTATED = "/camera/image_annotated"
ROS_TOPIC_PCB_RESULT = "/pcb_result"
ROS_TOPIC_INSPECTION_RESULT = "/aoi/inspection_result"
ROS_TOPIC_STATUS = "/aoi/status"
ROS_TOPIC_COMMAND = "/aoi/command"

ROS_PUBLISH_RAW_IMAGE = True
ROS_PUBLISH_ANNOTATED_IMAGE = True
ROS_PUBLISH_MAX_FPS = 15.0