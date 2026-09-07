import numpy as np

def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0: return []
    boxes, scores = np.array(boxes), np.array(scores)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1) * (y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2-xx1), np.maximum(0.0, yy2-yy1)
        ovr = (w*h) / (areas[i] + areas[order[1:]] - (w*h) + 1e-6)
        order = order[np.where(ovr <= iou_threshold)[0] + 1]
    return keep
