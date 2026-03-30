from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import numpy as np
import os
import threading
import tensorrt as trt

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_PATH = os.path.join(BASE_DIR, "best.engine")
TEST_IMAGE = os.path.join(BASE_DIR, "test.jpg")

CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open(ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

input_shape = engine.get_tensor_shape(input_name)
output_shape = engine.get_tensor_shape(output_name)

frame_global = None
stats = {cls: 0 for cls in CLASSES}
fps_value = 0

running = True
use_heatmap = True
DEV_MODE = False

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    DEV_MODE = True
    cap = None

heatmap = None


def safe_frame():
    img = cv2.imread(TEST_IMAGE)
    if img is None:
        img = np.zeros((640, 640, 3), dtype=np.uint8)
    return img


def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def nms(boxes, scores, iou_threshold=0.5):
    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        if len(idxs) == 1:
            break

        rest = idxs[1:]

        x1 = np.maximum(boxes[i][0], boxes[rest][:, 0])
        y1 = np.maximum(boxes[i][1], boxes[rest][:, 1])
        x2 = np.minimum(boxes[i][2], boxes[rest][:, 2])
        y2 = np.minimum(boxes[i][3], boxes[rest][:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area1 = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
        area2 = (boxes[rest][:, 2] - boxes[rest][:, 0]) * (boxes[rest][:, 3] - boxes[rest][:, 1])

        iou = inter / (area1 + area2 - inter + 1e-6)

        idxs = rest[iou < iou_threshold]

    return keep


def inference_loop():
    global frame_global, stats, fps_value, heatmap

    import pycuda.driver as cuda
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()

    d_input = cuda.mem_alloc(trt.volume(input_shape) * np.float32().nbytes)
    d_output = cuda.mem_alloc(trt.volume(output_shape) * np.float32().nbytes)

    while True:
        try:
            if not DEV_MODE and cap is not None:
                ret, frame = cap.read()
                if not ret:
                    frame = safe_frame()
            else:
                frame = safe_frame()

            orig_h, orig_w = frame.shape[:2]

            if heatmap is None:
                heatmap = np.zeros((orig_h, orig_w), dtype=np.float32)

            heatmap *= 0.95

            start = time.time()

            if running:
                img = preprocess(frame)

                cuda.memcpy_htod(d_input, img)
                context.execute_v2([int(d_input), int(d_output)])

                output = np.empty(output_shape, dtype=np.float32)
                cuda.memcpy_dtoh(output, d_output)

                pred = output[0].transpose(1, 0)

                boxes = []
                scores = []
                class_ids = []

                scale_x = orig_w / 640
                scale_y = orig_h / 640

                for row in pred:
                    obj = row[4]
                    if obj < 0.01:
                        continue

                    cls_id = int(np.argmax(row[5:]))
                    cls_score = row[5 + cls_id]
                    score = obj * cls_score

                    if score < 0.01:
                        continue

                    cx, cy, bw, bh = row[:4]

                    x1 = int((cx - bw / 2) * 640 * scale_x)
                    y1 = int((cy - bh / 2) * 640 * scale_y)
                    x2 = int((cx + bw / 2) * 640 * scale_x)
                    y2 = int((cy + bh / 2) * 640 * scale_y)

                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    class_ids.append(cls_id)

                stats = {cls: 0 for cls in CLASSES}

                if len(boxes) > 0:
                    boxes_np = np.array(boxes)
                    scores_np = np.array(scores)

                    keep = nms(boxes_np, scores_np)

                    for i in keep:
                        x1, y1, x2, y2 = boxes[i]
                        cls_id = class_ids[i]

                        label = CLASSES[cls_id]
                        stats[label] += 1

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {scores[i]:.2f}",
                                    (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 2)

                        if use_heatmap:
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2

                            y1h = max(0, cy - 10)
                            y2h = min(orig_h, cy + 10)
                            x1h = max(0, cx - 10)
                            x2h = min(orig_w, cx + 10)

                            heatmap[y1h:y2h, x1h:x2h] += 0.5

            fps_value = 1 / (time.time() - start + 1e-6)

            if use_heatmap:
                heat = cv2.applyColorMap(
                    np.clip(heatmap * 255, 0, 255).astype(np.uint8),
                    cv2.COLORMAP_JET
                )
                frame = cv2.addWeighted(frame, 0.7, heat, 0.3, 0)

            cv2.putText(frame, f"FPS: {fps_value:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if DEV_MODE:
                cv2.putText(frame, "DEV MODE", (orig_w - 180, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            frame_global = frame

        except Exception as e:
            print("loop error:", e)
            frame_global = safe_frame()

        time.sleep(0.01)

    ctx.pop()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    def gen():
        while True:
            img = frame_global if frame_global is not None else safe_frame()

            img = cv2.resize(img, (640, 640))

            ret, buffer = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            )

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")

            time.sleep(0.02)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def get_stats():
    return jsonify(stats)


@app.route("/status")
def get_status():
    return jsonify({"fps": round(fps_value, 2)})


@app.route("/control", methods=["POST"])
def control():
    global running, use_heatmap
    data = request.json

    if "run" in data:
        running = data["run"]

    if "heatmap" in data:
        use_heatmap = data["heatmap"]

    return jsonify({"ok": True})


if __name__ == "__main__":
    threading.Thread(target=inference_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)