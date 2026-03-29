from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("best.pt")

CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

cap = cv2.VideoCapture(0)

stats = {cls: 0 for cls in CLASSES}
fps_value = 0

# 控制開關
running = True
use_heatmap = True

heatmap = None


def generate():
    global stats, fps_value, heatmap, running, use_heatmap

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if not running:
            # 停止時顯示原畫面
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        h, w = frame.shape[:2]

        if heatmap is None:
            heatmap = np.zeros((h, w), dtype=np.float32)

        heatmap *= 0.95

        start = time.time()

        results = model(frame, imgsz=640, conf=0.3)[0]

        stats = {cls: 0 for cls in CLASSES}

        for box in results.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = CLASSES[cls_id]

            stats[label] += 1

            # heatmap
            if use_heatmap:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(heatmap, (cx, cy), 50, 10, -1)

            # 強化框
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),3)
            cv2.rectangle(frame, (x1, y1-25), (x2, y1), (0,255,0), -1)

            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1+5,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,0,0),2)

        fps_value = 1 / (time.time() - start)

        # heatmap overlay
        if use_heatmap:
            heat = cv2.applyColorMap(
                np.clip(heatmap,0,255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            frame = cv2.addWeighted(frame, 0.8, heat, 0.3, 0)

        # NG 警報
        if sum(stats.values()) > 0:
            cv2.putText(frame, "NG DETECTED",
                        (50,80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,(0,0,255),3)

        # FPS
        cv2.putText(frame, f"FPS: {fps_value:.2f}",
                    (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,255),2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def get_stats():
    return jsonify(stats)


@app.route('/status')
def get_status():
    return jsonify({
        "fps": round(fps_value,2),
        "running": running
    })


@app.route('/control', methods=['POST'])
def control():
    global running, use_heatmap

    data = request.json

    if "run" in data:
        running = data["run"]

    if "heatmap" in data:
        use_heatmap = data["heatmap"]

    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)