from pathlib import Path

import cv2
import numpy as np

from tooth_detector import detect_teeth
from classifier import classify_tooth

CONDITION_ORDER = ["Cavity", "Fillings", "Impacted", "Implant", "Normal"]
IMPLANT_CONDITION_ORDER = [
    "No Implant",
    "Stable Implant",
    "Peri-Implant Concern",
    "Implant Failure Risk",
]

BASE_DIR = Path(__file__).resolve().parent
MASK_DIR = BASE_DIR / "static" / "masks"
MASK_DIR.mkdir(parents=True, exist_ok=True)

COLOR_MAP = {
    "Cavity": (56, 56, 255),
    "Fillings": (255, 191, 0),
    "Impacted": (189, 60, 255),
    "Implant": (0, 160, 255),
    "Normal": (88, 186, 94),
}


def analyze_xray(image_path):
    image, regions = detect_teeth(image_path)
    analysis_id = Path(image_path).stem

    results = []
    for x, y, w, h, tooth in regions:
        label, conf = classify_tooth(tooth)
        anomaly_score, anomaly_flag = detect_structural_anomaly(tooth)
        implant_condition = classify_implant_condition(label, conf, anomaly_score)
        results.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "label": label,
                "confidence": round(conf * 100, 2),
                "anomaly_score": round(anomaly_score * 100, 2),
                "anomaly_flag": anomaly_flag,
                "implant_condition": implant_condition,
                "tooth_image": tooth,
            }
        )

    results = assign_fdi_numbers(results, image.shape)
    mask_name = build_segmentation_overlay(image, results, analysis_id)
    findings = summarize_findings(results, mask_name)

    for item in results:
        item.pop("tooth_image", None)

    return image, results, findings


def assign_fdi_numbers(results, image_shape):
    if not results:
        return results

    height = image_shape[0]
    center_split = height / 2.0

    for idx, item in enumerate(results):
        item["idx"] = idx
        item["cx"] = item["x"] + (item["w"] / 2.0)
        item["cy"] = item["y"] + (item["h"] / 2.0)
        item["fdi_number"] = None

    upper = [item for item in results if item["cy"] <= center_split]
    lower = [item for item in results if item["cy"] > center_split]

    upper_codes = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
    lower_codes = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
    assign_arch_codes(upper, upper_codes)
    assign_arch_codes(lower, lower_codes)

    for item in results:
        item.pop("idx", None)
        item.pop("cx", None)
        item.pop("cy", None)
    return results


def assign_arch_codes(arch_items, expected_codes):
    if not arch_items:
        return

    arch_items.sort(key=lambda item: item["cx"])
    min_x = arch_items[0]["cx"]
    max_x = arch_items[-1]["cx"]
    if max_x - min_x < 1:
        max_x = min_x + 1

    slot_positions = np.linspace(min_x, max_x, len(expected_codes))
    used_slots = set()

    for item in arch_items:
        distances = np.abs(slot_positions - item["cx"])
        nearest = np.argsort(distances)
        chosen_slot = None
        for slot in nearest:
            slot_id = int(slot)
            if slot_id not in used_slots:
                chosen_slot = slot_id
                break
        if chosen_slot is None:
            continue
        used_slots.add(chosen_slot)
        item["fdi_number"] = expected_codes[chosen_slot]


def detect_structural_anomaly(tooth_img):
    gray = cv2.cvtColor(tooth_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 10 or w < 10:
        return 0.0, False

    edges = cv2.Canny(gray, 40, 120)
    edge_density = float(np.count_nonzero(edges)) / float(h * w)

    left = gray[:, : w // 2]
    right = gray[:, w - (w // 2) :]
    right = cv2.flip(right, 1)
    min_w = min(left.shape[1], right.shape[1])
    if min_w > 0:
        symmetry_diff = np.mean(np.abs(left[:, :min_w].astype(np.float32) - right[:, :min_w].astype(np.float32))) / 255.0
    else:
        symmetry_diff = 0.0

    texture = np.std(gray.astype(np.float32)) / 128.0

    score = (0.45 * min(edge_density * 3.0, 1.0)) + (0.35 * min(symmetry_diff * 2.5, 1.0)) + (0.20 * min(texture, 1.0))
    flag = score >= 0.32
    return float(score), bool(flag)


def classify_implant_condition(label, confidence, anomaly_score):
    if label != "Implant":
        return "No Implant"
    if confidence >= 80 and anomaly_score < 0.25:
        return "Stable Implant"
    if anomaly_score < 0.42:
        return "Peri-Implant Concern"
    return "Implant Failure Risk"


def extract_tooth_mask(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 155:
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        fallback = np.zeros_like(gray, dtype=np.uint8)
        cv2.rectangle(fallback, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), 255, -1)
        return fallback

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask


def build_segmentation_overlay(image, results, analysis_id):
    overlay = image.copy()
    semantic = np.zeros_like(image)

    for item in results:
        x = item["x"]
        y = item["y"]
        w = item["w"]
        h = item["h"]
        tooth_crop = item["tooth_image"]
        local_mask = extract_tooth_mask(tooth_crop)
        color = COLOR_MAP.get(item["label"], (200, 200, 200))

        roi_semantic = semantic[y : y + h, x : x + w]
        roi_overlay = overlay[y : y + h, x : x + w]
        roi_semantic[local_mask > 0] = color
        roi_overlay[local_mask > 0] = (
            (0.35 * roi_overlay[local_mask > 0]) + (0.65 * np.array(color, dtype=np.float32))
        ).astype(np.uint8)

        fdi = item.get("fdi_number")
        if fdi is not None:
            cv2.putText(
                overlay,
                str(fdi),
                (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                str(fdi),
                (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

    blended = cv2.addWeighted(image, 0.55, semantic, 0.45, 0.0)
    blended = cv2.addWeighted(blended, 0.8, overlay, 0.2, 0.0)

    output_name = f"{analysis_id}_segmentation.png"
    output_path = MASK_DIR / output_name
    cv2.imwrite(str(output_path), blended)
    return output_name


def summarize_findings(results, mask_name):
    counts = {label: 0 for label in CONDITION_ORDER}
    implant_counts = {label: 0 for label in IMPLANT_CONDITION_ORDER}

    for item in results:
        label = item["label"]
        counts[label] = counts.get(label, 0) + 1
        implant_key = item["implant_condition"]
        implant_counts[implant_key] = implant_counts.get(implant_key, 0) + 1

    detected_teeth = len(results)
    expected_teeth = 32
    detected_fdi = sorted({item["fdi_number"] for item in results if item.get("fdi_number") is not None})
    expected_fdi = [
        18, 17, 16, 15, 14, 13, 12, 11,
        21, 22, 23, 24, 25, 26, 27, 28,
        48, 47, 46, 45, 44, 43, 42, 41,
        31, 32, 33, 34, 35, 36, 37, 38,
    ]
    missing_fdi = [code for code in expected_fdi if code not in detected_fdi]
    impacted_fdi = sorted([item["fdi_number"] for item in results if item.get("fdi_number") and item["label"] == "Impacted"])
    missing_or_impacted_fdi = sorted(set(missing_fdi + impacted_fdi))
    estimated_missing_teeth = len(missing_fdi) if detected_fdi else max(expected_teeth - detected_teeth, 0)

    abnormal_teeth = detected_teeth - counts.get("Normal", 0)
    anomaly_teeth = [item for item in results if item["anomaly_flag"]]

    top_findings = sorted(
        [item for item in results if item["label"] != "Normal"],
        key=lambda item: item["confidence"],
        reverse=True,
    )[:10]

    dominant_condition = "Normal"
    if abnormal_teeth > 0:
        dominant_condition = max(
            ["Cavity", "Fillings", "Impacted", "Implant"],
            key=lambda label: counts.get(label, 0),
        )

    detailed_summary = build_detailed_summary(
        detected_teeth=detected_teeth,
        expected_teeth=expected_teeth,
        estimated_missing_teeth=estimated_missing_teeth,
        counts=counts,
        abnormal_teeth=abnormal_teeth,
        dominant_condition=dominant_condition,
        missing_fdi=missing_fdi,
        missing_or_impacted_fdi=missing_or_impacted_fdi,
        anomaly_count=len(anomaly_teeth),
    )

    tooth_table = []
    for item in sorted(results, key=lambda row: (row.get("fdi_number") is None, row.get("fdi_number", 99), row["y"], row["x"])):
        tooth_table.append(
            {
                "fdi_number": item.get("fdi_number"),
                "label": item["label"],
                "confidence": item["confidence"],
                "anomaly_score": item["anomaly_score"],
                "anomaly_flag": item["anomaly_flag"],
                "implant_condition": item["implant_condition"],
                "x": item["x"],
                "y": item["y"],
                "w": item["w"],
                "h": item["h"],
            }
        )

    return {
        "detected_teeth": detected_teeth,
        "expected_teeth": expected_teeth,
        "estimated_missing_teeth": estimated_missing_teeth,
        "missing_fdi_numbers": missing_fdi,
        "impacted_fdi_numbers": impacted_fdi,
        "missing_or_impacted_fdi_numbers": missing_or_impacted_fdi,
        "abnormal_teeth": abnormal_teeth,
        "anomaly_teeth": len(anomaly_teeth),
        "dominant_condition": dominant_condition,
        "condition_counts": counts,
        "implant_condition_counts": implant_counts,
        "top_findings": top_findings,
        "detailed_summary": detailed_summary,
        "mask_image": mask_name,
        "tooth_table": tooth_table,
    }


def build_detailed_summary(
    detected_teeth,
    expected_teeth,
    estimated_missing_teeth,
    counts,
    abnormal_teeth,
    dominant_condition,
    missing_fdi,
    missing_or_impacted_fdi,
    anomaly_count,
):
    if abnormal_teeth == 0 and detected_teeth > 0:
        overall_status = "No major tooth-level abnormalities detected."
    elif detected_teeth == 0:
        overall_status = "Teeth were not segmented reliably in this OPG."
    else:
        overall_status = (
            f"{abnormal_teeth} teeth flagged as abnormal; dominant pattern is {dominant_condition.lower()}."
        )

    missing_preview = ", ".join(str(n) for n in missing_fdi[:12]) if missing_fdi else "None"
    if missing_fdi and len(missing_fdi) > 12:
        missing_preview = f"{missing_preview}, ..."

    missing_or_impacted_preview = ", ".join(str(n) for n in missing_or_impacted_fdi[:12]) if missing_or_impacted_fdi else "None"
    if missing_or_impacted_fdi and len(missing_or_impacted_fdi) > 12:
        missing_or_impacted_preview = f"{missing_or_impacted_preview}, ..."

    return {
        "overall_status": overall_status,
        "missing_teeth_info": f"Missing/undetected teeth: {estimated_missing_teeth} out of {expected_teeth}.",
        "missing_fdi_info": f"Missing FDI numbers: {missing_preview}",
        "missing_or_impacted_info": f"Absent or impacted FDI numbers: {missing_or_impacted_preview}",
        "cavity_info": f"Cavity-classified teeth: {counts.get('Cavity', 0)}.",
        "fillings_info": f"Fillings-classified teeth: {counts.get('Fillings', 0)}.",
        "impacted_info": f"Impacted teeth: {counts.get('Impacted', 0)}.",
        "implant_info": f"Implant teeth: {counts.get('Implant', 0)}.",
        "normal_info": f"Normal teeth: {counts.get('Normal', 0)}.",
        "anomaly_info": f"Structural anomalies detected: {anomaly_count}.",
    }
