import cv2


def detect_teeth(image_path):

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Unable to read input image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    image_area = h * w

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        4,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    min_area = max(250, int(0.00035 * image_area))
    max_area = int(0.08 * image_area)

    for c in contours:

        x, y, bw, bh = cv2.boundingRect(c)
        box_area = bw * bh
        aspect = bw / float(bh)

        if box_area < min_area or box_area > max_area:
            continue
        if bw < 18 or bh < 18:
            continue
        if aspect < 0.25 or aspect > 2.3:
            continue

        tooth = img[y : y + bh, x : x + bw]
        if tooth.size == 0:
            continue

        regions.append((x, y, bw, bh, tooth))

    regions = _suppress_overlaps(regions, iou_threshold=0.35)
    regions.sort(key=lambda region: (region[1], region[0]))

    if not regions:
        # Fallback region to keep pipeline operational when contour detection fails.
        pad_w = int(w * 0.08)
        pad_h = int(h * 0.12)
        x1 = max(pad_w, 0)
        y1 = max(pad_h, 0)
        x2 = min(w - pad_w, w)
        y2 = min(h - pad_h, h)
        if x2 > x1 and y2 > y1:
            tooth = img[y1:y2, x1:x2]
            if tooth.size > 0:
                regions.append((x1, y1, x2 - x1, y2 - y1, tooth))

    return img, regions


def _suppress_overlaps(regions, iou_threshold=0.35):
    if not regions:
        return []

    regions = sorted(regions, key=lambda r: r[2] * r[3], reverse=True)
    selected = []

    for candidate in regions:
        x1, y1, w1, h1, _ = candidate
        c_area = w1 * h1
        overlaps = False

        for accepted in selected:
            x2, y2, w2, h2, _ = accepted
            xx1 = max(x1, x2)
            yy1 = max(y1, y2)
            xx2 = min(x1 + w1, x2 + w2)
            yy2 = min(y1 + h1, y2 + h2)

            inter_w = max(0, xx2 - xx1)
            inter_h = max(0, yy2 - yy1)
            intersection = inter_w * inter_h
            if intersection == 0:
                continue

            union = c_area + (w2 * h2) - intersection
            iou = intersection / float(union)
            if iou >= iou_threshold:
                overlaps = True
                break

        if not overlaps:
            selected.append(candidate)

    return selected
