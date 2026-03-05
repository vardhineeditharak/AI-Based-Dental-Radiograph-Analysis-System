from pathlib import Path
import json
import uuid
from datetime import datetime
from collections.abc import Mapping, Sequence

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from analyzer import analyze_xray
from predictor import predict_image

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
REPORT_FOLDER = BASE_DIR / "static" / "reports"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["REPORT_FOLDER"] = str(REPORT_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.route("/")
def dashboard():
    return render_template("index.html")


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def save_report_file(report_name, payload):
    report_path = REPORT_FOLDER / report_name
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(to_builtin(payload), report_file, indent=2)
    return report_name


def to_builtin(value):
    if isinstance(value, Mapping):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


@app.route("/download-report/<report_name>")
def download_report(report_name):
    return send_from_directory(
        app.config["REPORT_FOLDER"],
        report_name,
        as_attachment=True,
        download_name=report_name,
    )


@app.route("/analyze", methods=["POST"])
def analyze():

    file = request.files.get("file")
    if file is None or not file.filename:
        return render_template("index.html", error="Select an image file to continue."), 400

    original_name = secure_filename(file.filename)
    if not original_name or not allowed_file(original_name):
        return render_template(
            "index.html",
            error="Unsupported file type. Use JPG, JPEG, PNG, or BMP.",
        ), 400

    ext = Path(original_name).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    filepath = UPLOAD_FOLDER / unique_name

    try:
        file.save(filepath)
        label, confidence = predict_image(str(filepath))
        _, tooth_results, findings = analyze_xray(str(filepath))
    except Exception:
        if filepath.exists():
            filepath.unlink()
        return render_template(
            "index.html",
            error="Analysis failed for this image. Please try another X-ray.",
        ), 500

    report_name = f"{Path(unique_name).stem}_report.json"
    report_payload = {
        "analysis_time_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "uploaded_image": unique_name,
        "image_level_prediction": {
            "label": label,
            "confidence_percent": round(confidence * 100, 2),
        },
        "findings": findings,
        "tooth_level_results": tooth_results,
    }
    save_report_file(report_name, report_payload)

    return render_template(
        "results.html",
        image=unique_name,
        label=label,
        confidence=round(confidence * 100, 2),
        findings=findings,
        report_name=report_name,
    )


if __name__ == "__main__":
    app.run(debug=True)
