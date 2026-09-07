from flask import Flask, Response, jsonify, render_template, request
import cv2
import time

from . import state

app = Flask(
    __name__,
    template_folder="../templates",
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():

    def gen():
        while True:

            time.sleep(0.04)

            if state.frame is None:
                continue

            _, buffer = cv2.imencode(
                ".jpg",
                state.frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    return Response(
        gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stats")
def stats():
    return jsonify(state.stats)


@app.route("/status")
def status():
    return jsonify({
        "fps": round(state.fps, 2)
    })


@app.route("/control", methods=["POST"])
def control():

    data = request.json

    if "run" in data:
        state.running = data["run"]

    return jsonify({
        "ok": True
    })