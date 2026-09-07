import os
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

from .config import (
    ENGINE_PATH,
    CLASSES,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
)
from .utils import nms


class TensorRTDetector:
    """TensorRT detector for Jetson with letterbox preprocessing and coordinate restoration."""

    _cuda_initialized = False

    def __init__(
        self,
        engine_path: str = ENGINE_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
    ):
        self.engine_path = engine_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.inference_time = 0.0

        self.logger = trt.Logger(trt.Logger.WARNING)

        if not TensorRTDetector._cuda_initialized:
            cuda.init()
            TensorRTDetector._cuda_initialized = True

        self.cuda_ctx = cuda.Device(0).make_context()
        self.cuda_ctx.push()

        try:
            self.stream = cuda.Stream()
            self.runtime = trt.Runtime(self.logger)

            if not os.path.exists(engine_path):
                raise FileNotFoundError(f"TensorRT engine not found at: {engine_path}")

            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())

            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine.")

            self.context = self.engine.create_execution_context()

            self.input_name = self.engine.get_tensor_name(0)
            self.output_name = self.engine.get_tensor_name(1)

            self.input_shape = self.engine.get_tensor_shape(self.input_name)
            self.output_shape = self.engine.get_tensor_shape(self.output_name)

            self.h_input = cuda.pagelocked_empty(
                trt.volume(self.input_shape),
                np.float32,
            )
            self.h_output = cuda.pagelocked_empty(
                trt.volume(self.output_shape),
                np.float32,
            )

            self.d_input = cuda.mem_alloc(self.h_input.nbytes)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        finally:
            self.cuda_ctx.pop()

    def set_thresholds(self, confidence: float, iou: float):
        """Update detection thresholds dynamically."""
        self.confidence_threshold = max(0.01, min(0.99, confidence))
        self.iou_threshold = max(0.01, min(0.99, iou))

    def _preprocess(self, frame: np.ndarray):
        """Letterbox resize frame with aspect ratio preservation and 114 padding."""
        orig_h, orig_w = frame.shape[:2]
        target_h, target_w = self.input_shape[2], self.input_shape[3]

        r = min(target_h / orig_h, target_w / orig_w)
        new_w = int(round(orig_w * r))
        new_h = int(round(orig_h * r))

        pad_x = (target_w - new_w) / 2.0
        pad_y = (target_h - new_h) / 2.0

        if orig_w != new_w or orig_h != new_h:
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame

        top = int(round(pad_y - 0.1))
        bottom = int(round(pad_y + 0.1))
        left = int(round(pad_x - 0.1))
        right = int(round(pad_x + 0.1))

        if top > 0 or bottom > 0 or left > 0 or right > 0:
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )
        else:
            padded = resized

        image = np.ascontiguousarray(
            padded[:, :, ::-1].transpose(2, 0, 1),
            dtype=np.float32,
        ) / 255.0

        np.copyto(self.h_input, image.ravel())
        return r, pad_x, pad_y, orig_w, orig_h

    def _infer(self):
        """Perform asynchronous inference on CUDA stream."""
        cuda.memcpy_htod_async(
            self.d_input,
            self.h_input,
            self.stream,
        )

        self.context.set_tensor_address(
            self.input_name,
            int(self.d_input),
        )

        self.context.set_tensor_address(
            self.output_name,
            int(self.d_output),
        )

        self.context.execute_async_v3(
            self.stream.handle,
        )

        cuda.memcpy_dtoh_async(
            self.h_output,
            self.d_output,
            self.stream,
        )

        self.stream.synchronize()

        return self.h_output.reshape(self.output_shape)

    def _postprocess(self, output, r, pad_x, pad_y, orig_w, orig_h):
        pred = output
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes_data = pred[:, :4]
        cls_scores = pred[:, 4:]

        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(len(pred)), cls_ids]

        mask = scores >= self.confidence_threshold
        if not np.any(mask):
            return [], {c: 0 for c in CLASSES}

        boxes_data = boxes_data[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]

        cx = boxes_data[:, 0]
        cy = boxes_data[:, 1]
        w = boxes_data[:, 2]
        h = boxes_data[:, 3]

        bx1 = (cx - w / 2.0 - pad_x) / r
        by1 = (cy - h / 2.0 - pad_y) / r
        bx2 = (cx + w / 2.0 - pad_x) / r
        by2 = (cy + h / 2.0 - pad_y) / r

        x1 = np.clip(np.round(bx1), 0, orig_w).astype(np.int32)
        y1 = np.clip(np.round(by1), 0, orig_h).astype(np.int32)
        x2 = np.clip(np.round(bx2), 0, orig_w).astype(np.int32)
        y2 = np.clip(np.round(by2), 0, orig_h).astype(np.int32)

        valid = (x2 > x1) & (y2 > y1)
        if not np.any(valid):
            return [], {c: 0 for c in CLASSES}

        x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
        scores = scores[valid]
        cls_ids = cls_ids[valid]

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = nms(boxes, scores, self.iou_threshold)

        detections = []
        stats = {c: 0 for c in CLASSES}

        for i in keep:
            cls_idx = int(cls_ids[i])
            label = CLASSES[cls_idx] if cls_idx < len(CLASSES) else f"class_{cls_idx}"
            stats[label] = stats.get(label, 0) + 1

            detections.append({
                "box": boxes[i].tolist(),
                "score": float(scores[i]),
                "label": label,
                "class_id": cls_idx,
            })

        return detections, stats

    def warmup(self, runs: int = 2):
        """Warm up engine with dummy image."""
        dummy = np.zeros(
            (self.input_shape[2], self.input_shape[3], 3),
            dtype=np.uint8,
        )
        for _ in range(runs):
            self.detect(dummy)

    def detect(self, frame: np.ndarray):
        """Run complete detection pipeline on input frame."""
        if frame is None or frame.size == 0:
            return {
                "detections": [],
                "stats": {c: 0 for c in CLASSES},
                "inference_ms": 0.0,
                "ok": True,
            }

        self.cuda_ctx.push()
        try:
            start = time.perf_counter()

            r, pad_x, pad_y, orig_w, orig_h = self._preprocess(frame)
            output = self._infer()
            detections, stats = self._postprocess(output, r, pad_x, pad_y, orig_w, orig_h)

            self.inference_time = (time.perf_counter() - start) * 1000.0

            ok = len(detections) == 0

            return {
                "detections": detections,
                "stats": stats,
                "inference_ms": self.inference_time,
                "ok": ok,
            }
        finally:
            self.cuda_ctx.pop()

    def close(self):
        """Release CUDA buffers and detach context."""
        try:
            if hasattr(self, "stream") and self.stream is not None:
                self.stream.synchronize()
        except Exception:
            pass

        for attr in ("d_input", "d_output"):
            try:
                obj = getattr(self, attr, None)
                if obj is not None:
                    obj.free()
            except Exception:
                pass

        self.context = None
        self.engine = None
        self.runtime = None

        try:
            if hasattr(self, "cuda_ctx") and self.cuda_ctx is not None:
                self.cuda_ctx.detach()
        except Exception:
            pass
