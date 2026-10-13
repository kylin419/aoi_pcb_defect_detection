import cv2
import numpy as np
from ..config import PCB_HSV_LOWER, PCB_HSV_UPPER

_SCALE = 0.25
_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # 縮圖後的尺寸
_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def pcb_green_area_detection(frame, return_info: bool = False):
    """
    回傳最大綠色（防焊）區域占整張畫面的比例。
    return_info=True 時回傳 dict：ratio / bbox(正規化) / center_offset / fill。
    """
    if frame is None or frame.size == 0:
        empty = {"ratio": 0.0, "bbox": None, "center_offset": None, "fill": 0.0}
        return empty if return_info else 0.0

    small = cv2.resize(frame, None, fx=_SCALE, fy=_SCALE,
                       interpolation=cv2.INTER_AREA)
    # A small blur suppresses sensor noise while retaining PCB edges.
    small = cv2.GaussianBlur(small, (5, 5), 0)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lower = np.array(PCB_HSV_LOWER, dtype=np.uint8)
    upper = np.array(PCB_HSV_UPPER, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN)    # 去雜點
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE)  # 補起零件、絲印造成的破洞

    h, w = mask.shape
    empty = {"ratio": 0.0, "bbox": None, "center_offset": None, "fill": 0.0}

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return empty if return_info else 0.0

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    ratio = area / float(h * w)
    if not return_info:
        return ratio

    x, y, bw, bh = cv2.boundingRect(c)
    return {
        "ratio": ratio,
        "bbox": (x / w, y / h, bw / w, bh / h),
        "center_offset": ((x + bw / 2) / w - 0.5, (y + bh / 2) / h - 0.5),
        "fill": area / float(max(1, bw * bh)),
    }
