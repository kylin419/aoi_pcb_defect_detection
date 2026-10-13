import sys
import cv2
from pcb_inspection.workers.localizer import pcb_green_area_detection

r0 = pcb_green_area_detection(cv2.imread(sys.argv[1]))
info = pcb_green_area_detection(cv2.imread(sys.argv[2]), return_info=True)
r1 = info["ratio"]
print(f"r0(空)={r0:.3f}  r1(板子)={r1:.3f}")
print(f"建議 PCB_MIN_RATIO = {r0 + 0.5*(r1-r0):.3f}")
print("bbox =", info["bbox"], " center_offset =",
      info["center_offset"], " fill =", f"{info['fill']:.2f}")
