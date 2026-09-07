import cv2
import numpy as np
from .config import CLASS_COLORS, CLASS_DISPLAY_NAMES


def draw_corner_rect(img, pt1, pt2, color, thickness=2, corner_len=12):
    """Draw a modern industrial rectangle with accented corners."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw main rectangle with slight transparency or thin line
    cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, thickness // 2))

    c_len = min(corner_len, (x2 - x1) // 3, (y2 - y1) // 3)
    if c_len > 2:
        th = thickness + 1
        # Top-left
        cv2.line(img, (x1, y1), (x1 + c_len, y1), color, th)
        cv2.line(img, (x1, y1), (x1, y1 + c_len), color, th)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - c_len, y1), color, th)
        cv2.line(img, (x2, y1), (x2, y1 + c_len), color, th)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + c_len, y2), color, th)
        cv2.line(img, (x1, y2), (x1, y2 - c_len), color, th)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - c_len, y2), color, th)
        cv2.line(img, (x2, y2), (x2, y2 - c_len), color, th)


def draw_detections(
    frame: np.ndarray,
    detections: list,
    inference_ms: float = None,
    fps: float = None,
    show_banner: bool = True,
) -> np.ndarray:
    """Draw professional AOI inspection overlay on image frame."""
    if frame is None:
        return frame

    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # Calculate responsive font scale based on resolution
    base_dim = max(h, w)
    font_scale = max(0.45, min(1.0, base_dim / 1500.0))
    line_thick = max(1, int(round(font_scale * 2)))

    defect_count = len(detections)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        raw_label = det["label"]
        score = det["score"]

        color = CLASS_COLORS.get(raw_label, (0, 255, 0))
        display_name = CLASS_DISPLAY_NAMES.get(raw_label, raw_label)
        caption = f"{display_name} {score:.0%}"

        # Draw accented box
        draw_corner_rect(overlay, (x1, y1), (x2, y2), color, thickness=line_thick + 1)

        # Draw label badge
        (tw, th), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thick
        )

        badge_y1 = max(0, y1 - th - 8)
        badge_y2 = y1
        if badge_y1 == 0:
            badge_y1 = y1
            badge_y2 = y1 + th + 8

        badge_x1 = max(0, x1)
        badge_x2 = min(w, x1 + tw + 10)

        # Filled badge background
        cv2.rectangle(
            overlay,
            (badge_x1, badge_y1),
            (badge_x2, badge_y2),
            color,
            -1,
        )

        # Label text in white/black for high contrast
        text_y = badge_y2 - 4 if badge_y1 == 0 else badge_y2 - 4
        cv2.putText(
            overlay,
            caption,
            (badge_x1 + 5, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            line_thick,
            cv2.LINE_AA,
        )

    # Top-left AOI status banner
    if show_banner:
        banner_w = int(240 * (base_dim / 1000.0))
        banner_w = max(220, min(banner_w, 360))
        banner_h = int(60 * (base_dim / 1000.0))
        banner_h = max(55, min(banner_h, 90))

        if defect_count == 0:
            status_text = "PASS [OK]"
            status_color = (0, 180, 0)
            sub_text = "No Defects Found"
        else:
            status_text = "FAIL [NG]"
            status_color = (0, 0, 220)
            sub_text = f"{defect_count} Defect{'s' if defect_count > 1 else ''} Detected"

        # Banner semi-transparent or solid rectangle
        cv2.rectangle(overlay, (12, 12), (12 + banner_w, 12 + banner_h), (25, 25, 25), -1)
        cv2.rectangle(overlay, (12, 12), (12 + banner_w, 12 + banner_h), status_color, 2)
        cv2.rectangle(overlay, (12, 12), (20, 12 + banner_h), status_color, -1)

        b_font_scale = font_scale * 1.1
        cv2.putText(
            overlay,
            status_text,
            (28, 12 + int(banner_h * 0.45)),
            cv2.FONT_HERSHEY_SIMPLEX,
            b_font_scale,
            status_color,
            line_thick + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            sub_text,
            (28, 12 + int(banner_h * 0.82)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.85,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    return overlay