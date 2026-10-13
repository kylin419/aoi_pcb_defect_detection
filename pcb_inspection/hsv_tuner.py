import sys
import time
import cv2
import numpy as np

if len(sys.argv) > 1:
    frame = cv2.imread(sys.argv[1])
else:
    from pcb_inspection.camera import Camera
    cam = Camera()
    ok, mode, msg = cam.open("csi")
    print(mode, msg)
    time.sleep(2.0)             
    ret, frame = cam.read()
    cam.release()
    cv2.imwrite("tuner_frame.jpg", frame)

frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
cv2.namedWindow("tuner")
for n, v, m in [("H lo", 35, 179), ("H hi", 90, 179), ("S lo", 40, 255), ("S hi", 255, 255), ("V lo", 30, 255), ("V hi", 255, 255)]:
    cv2.createTrackbar(n, "tuner", v, m, lambda x: None)

while True:
    def g(n): return cv2.getTrackbarPos(n, "tuner")
    lo = np.array([g("H lo"), g("S lo"), g("V lo")])
    hi = np.array([g("H hi"), g("S hi"), g("V hi")])
    mask = cv2.inRange(hsv, lo, hi)
    shown = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
    cv2.putText(shown, f"mask ratio={mask.mean()/255:.3f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("tuner", shown)
    k = cv2.waitKey(30) & 0xFF
    if k == ord('p'):
        print("LOWER =", tuple(lo), " UPPER =", tuple(hi))
    if k == ord('q'):
        break
cv2.destroyAllWindows()
