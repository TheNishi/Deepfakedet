"""EVIDENTIA — Flask Web UI for the Deepfake Detection Forensic Pipeline.

Serves the premium forensic dashboard and provides REST API endpoints for:
- Image forensic analysis with GradCAM visualization
- Video forensic analysis with keyframe extraction
- Batch file analysis
- PDF forensic report generation
"""

from __future__ import annotations

import base64
import io
import json
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from orchestration.forensics import (
    ModelBundle,
    load_forensics_models,
    run_forensic_analysis_image,
    run_forensic_analysis_video,
)

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = Path("config/inference.yaml")
EXPORT_DIR = Path("outputs") / "cam_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
MODEL_CACHE: list[ModelBundle] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tensor_to_rgb(tensor: torch.Tensor, *, normalize: bool) -> np.ndarray:
    """Convert a (C,H,W) tensor to an RGB float image in [0, 1]."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = tensor.detach().clone()
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=arr.dtype, device=arr.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=arr.dtype, device=arr.device)
        arr = arr * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
    arr = arr.clamp(0.0, 1.0)
    return arr.permute(1, 2, 0).cpu().numpy().astype(np.float32)


def _image_to_b64(img_arr: np.ndarray) -> str:
    """Convert numpy uint8 image array to base64 PNG string."""
    pil = Image.fromarray(img_arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _generate_gradcam_panels(image: Image.Image) -> list[dict]:
    """Generate GradCAM overlay panels for each model in the ensemble."""
    panels = []
    for bundle in MODEL_CACHE:
        tensor = bundle.transform(image)
        batch = tensor.unsqueeze(0).to(bundle.device)

        with torch.inference_mode():
            logits = bundle.model(batch)
            probs = F.softmax(logits, dim=1)
            cls_idx = int(probs.argmax(1))
            conf = float(probs[0, cls_idx] * 100.0)

        label = "REAL" if cls_idx == 1 else "FAKE"

        with GradCAM(model=bundle.model, target_layers=[bundle.target_layer]) as cam:
            grayscale = cam(
                input_tensor=batch,
                targets=[ClassifierOutputTarget(cls_idx)],
            )[0]

        rgb = _tensor_to_rgb(tensor, normalize=bundle.normalize)
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
        panels.append({
            "model": bundle.display_label,
            "label": label,
            "confidence": round(conf, 1),
            "image_b64": _image_to_b64(overlay),
        })

    return panels


def _generate_pdf_report(result: dict, cam_panels: list[dict], filename: str) -> io.BytesIO:
    """Generate a fully formatted forensic PDF report."""
    try:
        from fpdf import FPDF
    except ImportError:
        # Fallback: return a simple text-based PDF-like BytesIO
        buf = io.BytesIO()
        buf.write(b"%PDF-1.4\n% Simple text report (fpdf2 not installed)\n")
        buf.write(result.get("report", "No report").encode())
        buf.seek(0)
        return buf

    prediction = result.get("prediction", "UNKNOWN")
    confidence = result.get("model_confidence", 0.0)
    reliability = result.get("reliability_score", 0.0)
    visual = result.get("visual_consistency", "N/A")
    metadata = result.get("metadata_integrity", "N/A")
    artifacts = result.get("artifact_detection", "N/A")
    report_text = result.get("report", "")
    proofs = result.get("proofs", {})
    ts = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    is_fake = prediction.upper() == "FAKE"
    verdict_color = (220, 38, 38) if is_fake else (16, 185, 129)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # --- Header Bar ---
    pdf.set_fill_color(8, 20, 40)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_text_color(0, 219, 233)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "EVIDENTIA", ln=False)
    pdf.set_text_color(180, 200, 220)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 22)
    pdf.cell(0, 8, "Forensic Intelligence Platform  |  Digital Evidence Analysis Report")
    pdf.set_text_color(100, 130, 160)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_xy(10, 31)
    pdf.cell(0, 6, f"Generated: {ts}  |  Case File: {filename}")

    # --- Verdict Banner ---
    pdf.set_y(45)
    pdf.set_fill_color(*verdict_color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 18, f"  VERDICT: {prediction.upper()}", ln=True, fill=True, align="L")

    # --- Score Cards Row ---
    pdf.set_y(68)
    card_data = [
        ("MODEL CONFIDENCE", f"{confidence:.1f}%"),
        ("RELIABILITY SCORE", f"{reliability:.1f}%"),
        ("VISUAL CONSISTENCY", visual),
        ("METADATA INTEGRITY", metadata),
        ("ARTIFACT DETECTION", artifacts),
    ]
    card_w = 38
    card_x = 10
    for label, value in card_data:
        pdf.set_fill_color(17, 30, 55)
        pdf.set_draw_color(0, 100, 120)
        pdf.rect(card_x, pdf.get_y(), card_w, 22, "FD")
        pdf.set_text_color(100, 180, 210)
        pdf.set_font("Helvetica", "B", 6)
        pdf.set_xy(card_x + 1, pdf.get_y() + 2)
        pdf.cell(card_w - 2, 5, label, align="C")
        pdf.set_text_color(220, 240, 255)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_xy(card_x + 1, pdf.get_y() + 6)
        pdf.cell(card_w - 2, 8, str(value), align="C")
        card_x += card_w + 2

    # --- Reliability Score Bar ---
    pdf.set_y(96)
    pdf.set_text_color(0, 219, 233)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "RELIABILITY SCORE", ln=True)
    bar_x, bar_y = 10, pdf.get_y()
    bar_total_w = 190
    bar_h = 8
    # Background bar
    pdf.set_fill_color(30, 45, 70)
    pdf.rect(bar_x, bar_y, bar_total_w, bar_h, "F")
    # Filled portion
    fill_w = max(2, int(bar_total_w * reliability / 100))
    if reliability >= 70:
        pdf.set_fill_color(16, 185, 129)
    elif reliability >= 40:
        pdf.set_fill_color(245, 158, 11)
    else:
        pdf.set_fill_color(220, 38, 38)
    pdf.rect(bar_x, bar_y, fill_w, bar_h, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_xy(bar_x + fill_w - 20, bar_y + 1)
    pdf.cell(18, 6, f"{reliability:.1f}%", align="R")
    pdf.set_y(bar_y + bar_h + 4)

    # --- Forensic Analysis Section ---
    pdf.set_text_color(0, 219, 233)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "FORENSIC ANALYSIS REPORT", ln=True)
    pdf.set_draw_color(0, 100, 120)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.set_y(pdf.get_y() + 3)

    pdf.set_fill_color(10, 18, 35)
    pdf.set_draw_color(30, 60, 90)
    box_y = pdf.get_y()
    # Write report text in code-style box
    pdf.set_font("Courier", "", 8)
    pdf.set_text_color(100, 240, 180)
    for line in report_text.split("\n"):
        pdf.set_x(12)
        pdf.cell(0, 5, line.rstrip(), ln=True)
    report_box_h = pdf.get_y() - box_y + 4
    pdf.set_fill_color(10, 18, 35)
    pdf.rect(10, box_y - 2, 190, report_box_h, "FD")

    # --- Digital Evidence Proofs ---
    if proofs:
        pdf.set_y(pdf.get_y() + 5)
        pdf.set_text_color(0, 219, 233)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "DIGITAL EVIDENCE & PROOFS", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.set_y(pdf.get_y() + 3)

        proof_items = []
        if "metadata_tags" in proofs and proofs["metadata_tags"]:
            proof_items.append(("Suspicious Metadata Tags", ", ".join(proofs["metadata_tags"])))
        if "high_freq_ratio" in proofs:
            proof_items.append(("High Freq. Artifact Ratio", f"{proofs['high_freq_ratio']:.4f}"))
        if "noise_std" in proofs:
            proof_items.append(("Sensor Noise Std Dev", f"{proofs['noise_std']:.4f}"))
        if "texture_inconsistency" in proofs:
            proof_items.append(("Texture Inconsistency Score", f"{proofs['texture_inconsistency']:.4f}"))
        if "temporal_variance" in proofs:
            proof_items.append(("Temporal Frame Variance", f"{proofs['temporal_variance']:.4f}"))

        col_w = 93
        for i, (label, value) in enumerate(proof_items):
            col_x = 10 if i % 2 == 0 else 107
            row_y = pdf.get_y()
            pdf.set_fill_color(17, 30, 55)
            pdf.set_draw_color(0, 80, 100)
            pdf.rect(col_x, row_y, col_w, 16, "FD")
            pdf.set_text_color(100, 170, 210)
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_xy(col_x + 2, row_y + 2)
            pdf.cell(col_w - 4, 5, label.upper())
            pdf.set_text_color(220, 240, 255)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_xy(col_x + 2, row_y + 8)
            pdf.cell(col_w - 4, 6, str(value))
            if i % 2 == 1:
                pdf.set_y(row_y + 18)

        if len(proof_items) % 2 == 1:
            pdf.set_y(pdf.get_y() + 18)

    # --- GradCAM Panels ---
    if cam_panels:
        pdf.add_page()
        pdf.set_fill_color(8, 20, 40)
        pdf.rect(0, 0, 210, 15, "F")
        pdf.set_text_color(0, 219, 233)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_xy(10, 4)
        pdf.cell(0, 7, "GRAD-CAM SPATIAL ATTRIBUTION MAPS — Model Ensemble Evidence")
        pdf.set_y(20)

        panel_w = 58
        panel_h = 58
        x_positions = [10, 75, 140]
        for i, panel in enumerate(cam_panels[:3]):
            px = x_positions[i % 3]
            py = pdf.get_y()
            # Panel label
            pdf.set_fill_color(17, 30, 55)
            pdf.rect(px, py, panel_w, 8, "F")
            pdf.set_text_color(0, 219, 233)
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_xy(px + 1, py + 1)
            verdict_str = f"{panel['label']} ({panel['confidence']}%)"
            pdf.cell(panel_w - 2, 6, f"{panel['model']}  |  {verdict_str}", align="C")

            # Embed CAM image
            try:
                img_data = base64.b64decode(panel["image_b64"])
                img_buf = io.BytesIO(img_data)
                tmp_path = EXPORT_DIR / f"_pdf_cam_{i}.png"
                with open(tmp_path, "wb") as f_out:
                    f_out.write(img_data)
                pdf.image(str(tmp_path), x=px, y=py + 9, w=panel_w, h=panel_h)
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

            if i % 3 == 2:
                pdf.set_y(py + panel_h + 12)

    # --- Chain of Custody Footer ---
    pdf.set_y(-35)
    pdf.set_fill_color(8, 20, 40)
    pdf.rect(0, pdf.get_y() - 2, 210, 40, "F")
    pdf.set_draw_color(0, 100, 120)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.set_y(pdf.get_y() + 2)
    pdf.set_text_color(100, 160, 200)
    pdf.set_font("Helvetica", "B", 7)
    pdf.cell(0, 5, "CHAIN OF CUSTODY — EVIDENTIA Forensic Intelligence Platform", ln=True, align="C")
    pdf.set_text_color(80, 120, 160)
    pdf.set_font("Helvetica", "", 6)
    pdf.cell(0, 4, f"Report ID: EVD-{datetime.now(tz=UTC).strftime('%Y%m%d%H%M%S')}  |  Analysis Engine: EfficientNet-B3 + FasterViT-2-224 + EfficientFormerV2-S1  |  {ts}", align="C", ln=True)
    pdf.cell(0, 4, "This report is generated by an automated AI forensic system. Results should be reviewed by a qualified forensic examiner.", align="C")

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze/image", methods=["POST"])
def analyze_image():
    """Run full forensic analysis on an uploaded image."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    try:
        result = run_forensic_analysis_image(image, MODEL_CACHE)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {e}"}), 500

    # Generate GradCAM panels
    cam_panels = []
    try:
        cam_panels = _generate_gradcam_panels(image)
    except Exception:
        traceback.print_exc()

    return jsonify({
        "prediction": result["prediction"],
        "model_confidence": round(result["model_confidence"], 2),
        "reliability_score": round(result["reliability_score"], 2),
        "visual_consistency": result["visual_consistency"],
        "metadata_integrity": result["metadata_integrity"],
        "artifact_detection": result["artifact_detection"],
        "report": result["report"],
        "proofs": result.get("proofs", {}),
        "cam_panels": cam_panels,
        "filename": file.filename,
    })


@app.route("/api/analyze/video", methods=["POST"])
def analyze_video():
    """Run full forensic analysis on an uploaded video file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Save temp file
    tmp_path = EXPORT_DIR / f"tmp_video_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S%f')}{Path(file.filename).suffix}"
    try:
        file.save(str(tmp_path))
        result = run_forensic_analysis_video(tmp_path, MODEL_CACHE)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {e}"}), 500
    finally:
        tmp_path.unlink(missing_ok=True)

    # Generate GradCAM for up to 4 keyframes
    keyframe_panels = []
    keyframes = result.get("keyframes", [])
    if keyframes:
        indices = np.linspace(0, len(keyframes) - 1, min(len(keyframes), 4), dtype=int)
        for idx in indices:
            try:
                frame_panels = _generate_gradcam_panels(keyframes[idx])
                if frame_panels:
                    kf = frame_panels[0]  # Primary model panel
                    kf["frame_idx"] = int(idx)
                    keyframe_panels.append(kf)
            except Exception:
                pass

    return jsonify({
        "prediction": result["prediction"],
        "model_confidence": round(result["model_confidence"], 2),
        "reliability_score": round(result["reliability_score"], 2),
        "visual_consistency": result["visual_consistency"],
        "metadata_integrity": result["metadata_integrity"],
        "artifact_detection": result["artifact_detection"],
        "report": result["report"],
        "proofs": result.get("proofs", {}),
        "keyframe_panels": keyframe_panels,
        "filename": file.filename,
    })


@app.route("/api/analyze/batch", methods=["POST"])
def analyze_batch():
    """Run forensic analysis on multiple uploaded files."""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    results = []
    for file in files:
        if not file.filename:
            continue
        suffix = Path(file.filename).suffix.lower()
        image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

        try:
            if suffix in image_suffixes:
                image = Image.open(file.stream).convert("RGB")
                result = run_forensic_analysis_image(image, MODEL_CACHE)
                results.append({
                    "filename": file.filename,
                    "type": "image",
                    "prediction": result["prediction"],
                    "model_confidence": round(result["model_confidence"], 2),
                    "reliability_score": round(result["reliability_score"], 2),
                    "visual_consistency": result["visual_consistency"],
                    "metadata_integrity": result["metadata_integrity"],
                    "artifact_detection": result["artifact_detection"],
                })
            elif suffix in video_suffixes:
                tmp_path = EXPORT_DIR / f"tmp_batch_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S%f')}{suffix}"
                file.save(str(tmp_path))
                try:
                    result = run_forensic_analysis_video(tmp_path, MODEL_CACHE)
                    results.append({
                        "filename": file.filename,
                        "type": "video",
                        "prediction": result["prediction"],
                        "model_confidence": round(result["model_confidence"], 2),
                        "reliability_score": round(result["reliability_score"], 2),
                        "visual_consistency": result["visual_consistency"],
                        "metadata_integrity": result["metadata_integrity"],
                        "artifact_detection": result["artifact_detection"],
                    })
                finally:
                    tmp_path.unlink(missing_ok=True)
            else:
                results.append({
                    "filename": file.filename,
                    "type": "unknown",
                    "prediction": "SKIPPED",
                    "error": f"Unsupported format: {suffix}",
                })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "type": "error",
                "prediction": "ERROR",
                "error": str(e),
            })

    return jsonify({"results": results})


@app.route("/api/report/pdf", methods=["POST"])
def generate_pdf():
    """Generate and return a PDF forensic report."""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    result = data.get("result", {})
    cam_panels = data.get("cam_panels", [])
    filename = data.get("filename", "evidence")

    try:
        pdf_buf = _generate_pdf_report(result, cam_panels, filename)
        report_name = f"EVIDENTIA_Report_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(
            pdf_buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=report_name,
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"PDF generation failed: {e}"}), 500


@app.route("/api/analyze/audio", methods=["POST"])
def analyze_audio():
    """Mock endpoint for Audio Forensics."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    
    # Mock data to match the UI image
    return jsonify({
        "prediction": "REAL",
        "model_confidence": 98.2,
        "reliability_score": 87.6,
        "filename": file.filename,
        "proofs": {
            "Voice Clone Probability": "23%",
            "Speaker Verification": "Match",
            "Deepfake Audio Detection": "Low",
            "Noise Manipulation": "Not Detected",
            "Audio Integrity": "High"
        }
    })

@app.route("/api/analyze/text", methods=["POST"])
def analyze_text():
    """Mock endpoint for Text Evidence Analysis."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    
    # Mock data to match the UI image
    return jsonify({
        "prediction": "REAL",
        "model_confidence": 95.0,
        "reliability_score": 89.0,
        "filename": file.filename,
        "proofs": {
            "Sentiment": "Neutral",
            "Threat Detection": "Low",
            "Toxicity": "Low",
            "Fake News Indicator": "Medium",
            "AI Generated Probability": "18%"
        }
    })

@app.route("/api/dashboard/stats", methods=["GET"])
def dashboard_stats():
    """Mock endpoint for Dashboard Stats top bar."""
    return jsonify({
        "evidence_processed": "2,457",
        "evidence_processed_change": "+18%",
        "reliability_avg": "87.6",
        "threat_level": "LOW",
        "court_admissibility": "HIGH",
        "active_cases": "23"
    })

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def initialize(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Load models from config."""
    global MODEL_CACHE
    print(f"[EVIDENTIA] Loading forensic models from {config_path}...")
    try:
        MODEL_CACHE = load_forensics_models(config_path)
        print(f"[EVIDENTIA] Loaded {len(MODEL_CACHE)} model(s) successfully.")
    except Exception as e:
        print(f"[EVIDENTIA] Warning: Could not load models: {e}")
        print("[EVIDENTIA] Dashboard will start — upload will fail until models are available.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EVIDENTIA Forensic Dashboard")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    initialize(args.config)
    app.run(host=args.host, port=args.port, debug=args.debug)
