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


# -------------------- utils --------------------

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


def nms(boxes, scores, iou_threshold=0.45):
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


# -------------------- inference --------------------

def inference_loop():
    global frame_global, stats, fps_value, heatmap

    import pycuda.driver as cuda
    cuda.init()
    ctx = cuda.Device(0).make_context()

    runtime = trt.Runtime(TRT_LOGGER)
    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    input_name, output_name = None, None

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name

    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)

    import pycuda.driver as cuda
    d_input = cuda.mem_alloc(trt.volume(input_shape) * np.float32().nbytes)
    d_output = cuda.mem_alloc(trt.volume(output_shape) * np.float32().nbytes)

    stream = cuda.Stream()

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

            heatmap *= 0.9

            start = time.time()

            if running:
                img = preprocess(frame)

                cuda.memcpy_htod_async(d_input, img, stream)

                context.set_tensor_address(input_name, int(d_input))
                context.set_tensor_address(output_name, int(d_output))
                context.execute_async_v3(stream.handle)

                output = np.empty(output_shape, dtype=np.float32)
                cuda.memcpy_dtoh_async(output, d_output, stream)
                stream.synchronize()

                print("MAX:", np.max(output))

                # 🔥 正確 decode（關鍵）
                pred = output[0].transpose(1, 0)

                boxes, scores, class_ids = [], [], []

                for row in pred:
                    cx, cy, w, h = row[:4]

                    obj = row[4]          
                    cls_scores = row[5:]

                    cls_id = int(np.argmax(cls_scores))
                    cls_score = cls_scores[cls_id]

                    score = obj * cls_score

                    if score < 0.4:
                        continue

                    # bbox
                    x1 = int((cx - w / 2) * orig_w / 640)
                    y1 = int((cy - h / 2) * orig_h / 640)
                    x2 = int((cx + w / 2) * orig_w / 640)
                    y2 = int((cy + h / 2) * orig_h / 640)

                    x1 = max(0, min(orig_w - 1, x1))
                    y1 = max(0, min(orig_h - 1, y1))
                    x2 = max(0, min(orig_w - 1, x2))
                    y2 = max(0, min(orig_h - 1, y2))

            
                    if (x2 - x1) < 8 or (y2 - y1) < 8:
                        continue
                    if (x2 - x1) * (y2 - y1) > 20000:
                        continue

                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    class_ids.append(cls_id)

                stats = {cls: 0 for cls in CLASSES}

                if len(boxes) > 0:
                    keep = nms(np.array(boxes), np.array(scores))
                    keep = keep[:10]

                    for i in keep:
                        x1, y1, x2, y2 = boxes[i]
                        label = CLASSES[class_ids[i]]

                        stats[label] += 1

                       
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, f"{label} {scores[i]:.2f}",
                                    (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 0, 255), 2)

                        if use_heatmap:
                            cxm = (x1 + x2) // 2
                            cym = (y1 + y2) // 2
                            heatmap[max(0, cym-5):cym+5, max(0, cxm-5):cxm+5] += 0.3

            fps_value = 1 / (time.time() - start + 1e-6)

            if use_heatmap:
                heat = cv2.applyColorMap(
                    np.clip(heatmap * 255, 0, 255).astype(np.uint8),
                    cv2.COLORMAP_JET
                )
                frame = cv2.addWeighted(frame, 0.85, heat, 0.15, 0)

            cv2.putText(frame, f"FPS: {fps_value:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if DEV_MODE:
                cv2.putText(frame, "DEV MODE", (orig_w - 180, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            frame_global = frame

        except Exception as e:
            print("ERROR:", e)
            frame_global = safe_frame()

        time.sleep(0.03)

    ctx.pop()


# -------------------- routes --------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    def gen():
        while True:
            img = frame_global if frame_global is not None else safe_frame()
            img = cv2.resize(img, (640, 640))
            _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")
            time.sleep(0.03)

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


# -------------------- main --------------------

if __name__ == "__main__":
    threading.Thread(target=inference_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)