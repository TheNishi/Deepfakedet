"""Web UI for Real vs Fake face detection with Grad-CAM visualization.

This script loads three trained models (EfficientNet-B3, FasterViT-2-224,
EfficientFormerV2-S1), performs inference on an input image or video,
performs detailed forensic checks (visual consistency, metadata integrity,
frequency domain anomalies), computes a Reliability Score, and displays
the report and Grad-CAM explanations via a premium Gradio interface.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as f
from PIL import Image, ImageDraw, ImageFont
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from orchestration.forensics import (
    ModelBundle,
    load_forensics_models,
    run_forensic_analysis_image,
    run_forensic_analysis_video,
)

# ---------------------------------------------------------------------
# Device, config, and export settings
# ---------------------------------------------------------------------
DEFAULT_CONFIG_PATH = Path("config/inference.yaml")

EXPORT_SCALE = 2
EXPORT_DIR = Path("outputs") / "cam_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CACHE: list[ModelBundle] = []


def _tensor_to_rgb(tensor: torch.Tensor, *, normalize: bool) -> np.ndarray:
    """Convert a (C,H,W) tensor to an RGB float image in [0, 1]."""
    if tensor.ndim == 4:
        if tensor.size(0) != 1:
            msg = "Expected batch of size 1 for visualization."
            raise ValueError(msg)
        tensor = tensor[0]

    if tensor.ndim != 3:
        msg = "Expected a 3D tensor for visualization."
        raise ValueError(msg)

    arr = tensor.detach().clone()
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=arr.dtype, device=arr.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=arr.dtype, device=arr.device)
        arr = arr * std.view(-1, 1, 1) + mean.view(-1, 1, 1)

    arr = arr.clamp(0.0, 1.0)
    rgb = arr.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return rgb


def _add_label(img_rgb_uint8: np.ndarray, text: str) -> np.ndarray:
    """Overlay a readable text label onto the top-left of an image."""
    img = Image.fromarray(img_rgb_uint8)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text(
        (6, 6),
        text,
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
        font=font,
    )
    return np.asarray(img)


def initialize_from_config(config_path: Path) -> None:
    """Load orchestrator config and populate inference resources using the forensics loader."""
    global MODEL_CACHE
    MODEL_CACHE = load_forensics_models(config_path)


# ---------------------------------------------------------------------
# Forensic Analysis Callbacks
# ---------------------------------------------------------------------
def format_proofs_html(report: str, proofs: dict) -> str:
    html = f"<div class='report-output'><pre>{report}</pre></div>"

    if proofs:
        html += "<div class='proofs-container'>"
        html += "<h3 class='proofs-title'>Digital Evidence & Proofs</h3>"
        html += "<div class='proof-grid'>"

        tags = proofs.get("metadata_tags", [])
        tags_str = ", ".join(tags) if tags else "None detected"
        html += f"""
        <div class='proof-card'>
            <div class='proof-label'>Suspicious Metadata Tags</div>
            <div class='proof-value'>{tags_str}</div>
        </div>"""

        if "high_freq_ratio" in proofs:
            html += f"""
            <div class='proof-card'>
                <div class='proof-label'>High Frequency Artifact Ratio</div>
                <div class='proof-value'>{proofs['high_freq_ratio']:.4f}</div>
            </div>"""

        if "noise_std" in proofs:
            html += f"""
            <div class='proof-card'>
                <div class='proof-label'>Sensor Noise Std Dev</div>
                <div class='proof-value'>{proofs['noise_std']:.4f}</div>
            </div>"""

        if "texture_inconsistency" in proofs:
            html += f"""
            <div class='proof-card'>
                <div class='proof-label'>Texture Inconsistency Score</div>
                <div class='proof-value'>{proofs['texture_inconsistency']:.4f}</div>
            </div>"""

        if "temporal_variance" in proofs:
            html += f"""
            <div class='proof-card'>
                <div class='proof-label'>Temporal Frame Variance</div>
                <div class='proof-value'>{proofs['temporal_variance']:.4f}</div>
            </div>"""

        html += "</div></div>"
    return html

def process_image(image: Image.Image | None) -> tuple[np.ndarray | None, str]:
    """Run ensembled prediction and forensic checks on a PIL Image."""
    if image is None:
        return None, "<div class='report-output'>Error: No image provided.</div>"

    # 1. Run ensembled forensic analysis
    res = run_forensic_analysis_image(image, MODEL_CACHE)
    report_html = format_proofs_html(res["report"], res.get("proofs", {}))

    # 2. Generate side-by-side Grad-CAM visualizations for each active model
    panels = []
    for bundle in MODEL_CACHE:
        tensor = bundle.transform(image)
        batch = tensor.unsqueeze(0).to(bundle.device)

        with torch.inference_mode():
            logits = bundle.model(batch)
            probs = f.softmax(logits, dim=1)
            cls_idx = int(probs.argmax(1))
            conf = float(probs[0, cls_idx] * 100.0)

        label = "REAL" if cls_idx == 1 else "FAKE"

        # Generate Grad-CAM for this model
        with GradCAM(model=bundle.model, target_layers=[bundle.target_layer]) as cam:
            grayscale = cam(
                input_tensor=batch,
                targets=[ClassifierOutputTarget(cls_idx)],
            )[0]

        rgb = _tensor_to_rgb(tensor, normalize=bundle.normalize)
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
        panel = _add_label(overlay, f"{bundle.display_label} {label} ({conf:.1f}%)")
        panels.append(panel)

    if not panels:
        return None, report_html

    side_by_side = np.concatenate(panels, axis=1)

    # Export high-res composite panel
    h, w, _ = side_by_side.shape
    export_img = Image.fromarray(side_by_side).resize(
        (w * EXPORT_SCALE, h * EXPORT_SCALE),
        resample=Image.BICUBIC,
    )
    out_path = (
        EXPORT_DIR
        / f"cam_triptych_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}.png"
    )
    export_img.save(out_path, format="PNG", optimize=True)

    return np.asarray(export_img), report_html


def process_video(video_path: str | None) -> tuple[list[np.ndarray], str]:
    """Run ensembled prediction and forensic checks on a video, producing keyframe visualizations."""
    if not video_path:
        return [], "<div class='report-output'>Error: No video file provided.</div>"

    res = run_forensic_analysis_video(Path(video_path), MODEL_CACHE)
    keyframes = res.get("keyframes", [])
    report_html = format_proofs_html(res.get("report", ""), res.get("proofs", {}))

    if not keyframes:
        return [], report_html

    cam_overlays = []
    # Generate Grad-CAM for up to 4 keyframes to avoid excessive CPU/GPU lag
    num_to_cam = min(len(keyframes), 4)
    indices = np.linspace(0, len(keyframes) - 1, num_to_cam, dtype=int)

    for idx in indices:
        img = keyframes[idx]
        if MODEL_CACHE:
            bundle = MODEL_CACHE[0]  # Show Grad-CAM using the primary model
            tensor = bundle.transform(img)
            batch = tensor.unsqueeze(0).to(bundle.device)

            with torch.inference_mode():
                logits = bundle.model(batch)
                probs = f.softmax(logits, dim=1)
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
            panel = _add_label(overlay, f"Keyframe {idx} ({label} {conf:.1f}%)")
            cam_overlays.append(panel)

    return cam_overlays, report_html


# ---------------------------------------------------------------------
# Gradio UI Layout Dashboard
# ---------------------------------------------------------------------
def build_interface(config_path: Path = DEFAULT_CONFIG_PATH) -> gr.Blocks:
    """Create a Gradio interface styled as a premium forensics dashboard."""
    initialize_from_config(config_path)

    css = """
    .gradio-container {
        background-color: #0b0f19 !important;
        color: #e5e7eb !important;
        font-family: 'Outfit', 'Inter', system-ui, sans-serif !important;
    }
    .header-box {
        text-align: center;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.1) 0%, rgba(13, 148, 136, 0.1) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(8px);
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
        letter-spacing: -0.025em;
    }
    .header-subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        margin: 0;
    }
    .forensic-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 16px;
    }
    .run-btn {
        background: linear-gradient(135deg, #2563eb 0%, #0d9488 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .run-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4) !important;
    }
    .report-output pre {
        font-family: 'Courier New', Courier, monospace !important;
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
        background-color: #030712 !important;
        border: 1px solid #1f2937 !important;
        color: #10b981 !important;
        font-weight: 600 !important;
        padding: 16px;
        border-radius: 8px;
        white-space: pre-wrap;
        margin: 0;
    }
    .proofs-container {
        margin-top: 20px;
    }
    .proofs-title {
        color: #60a5fa;
        margin-bottom: 12px;
        font-size: 1.2rem;
        font-weight: 700;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 8px;
    }
    .proof-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
    }
    .proof-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .proof-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(96, 165, 250, 0.2);
    }
    .proof-label {
        color: #9ca3af;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .proof-value {
        color: #f3f4f6;
        font-size: 1.1rem;
        font-weight: 600;
        word-break: break-word;
    }
    """

    with gr.Blocks(css=css, theme=gr.themes.Default()) as demo:
        gr.HTML("""
        <div class='header-box'>
            <h1 class='header-title'>AI Digital Evidence Analysis System</h1>
            <p class='header-subtitle'>Forensic authenticity & integrity evaluation for images and videos using deep learning ensembles.</p>
        </div>
        """)

        with gr.Tabs():
            with gr.Tab("Image Forensic Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<div class='forensic-card'><h3>Input Image</h3>Upload a digital photo or screenshot to evaluate its authenticity.</div>")
                        image_input = gr.Image(type="pil", label="Source Image")
                        image_btn = gr.Button("Perform Forensic Audit", elem_classes="run-btn")

                    with gr.Column(scale=1):
                        gr.HTML("<div class='forensic-card'><h3>Grad-CAM Attribution Map</h3>Highlights regions of localized manipulation or texture anomalies detected by the model ensemble.</div>")
                        image_output_cam = gr.Image(type="numpy", label="Grad-CAM Spatial Evidence Attribution")

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.HTML("<div class='forensic-card'><h3>Audit Report</h3>Structured verification and digital forensics ledger.</div>")
                        image_report = gr.HTML(
                            elem_classes="report-container"
                        )

            with gr.Tab("Video Forensic Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<div class='forensic-card'><h3>Input Video</h3>Upload a digital video to scan across multiple keyframes for temporal synthesis markers.</div>")
                        video_input = gr.Video(label="Source Video")
                        video_btn = gr.Button("Perform Forensic Audit", elem_classes="run-btn")

                    with gr.Column(scale=1):
                        gr.HTML("<div class='forensic-card'><h3>Keyframe Visual Analysis</h3>Grad-CAM verification layers generated across sampled video frames.</div>")
                        video_gallery = gr.Gallery(label="Keyframe Grad-CAM Explanations", columns=2, height="auto")

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.HTML("<div class='forensic-card'><h3>Audit Report</h3>Structured verification and digital forensics ledger.</div>")
                        video_report = gr.HTML(
                            elem_classes="report-container"
                        )

        image_btn.click(
            fn=process_image,
            inputs=[image_input],
            outputs=[image_output_cam, image_report]
        )

        video_btn.click(
            fn=process_video,
            inputs=[video_input],
            outputs=[video_gallery, video_report]
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deepfake detection UI")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to an orchestrator inference YAML config.",
    )
    args = parser.parse_args()

    demo_app = build_interface(args.config)
    demo_app.launch()

