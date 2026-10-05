import cv2
import numpy as np

def pcb_green_area_detection(frame)->float:
    """
    計算畫面中最大 PCB 綠色區域所占的比例。

    回傳:
        float -> green_ratio
                 0.0 表示沒有偵測到綠色 PCB 區域
    """
    image_area = frame.shape[0] * frame.shape[1]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([55, 140, 30])
    upper = np.array([75, 255, 130])

    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

    if not contours:
            return 0.0

    largest_contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest_contour)

    green_ratio = area / image_area

    print(f"PCB Green Area: {area:.0f}")
    print(f"Green Ratio: {green_ratio:.3f}")

    return green_ratio