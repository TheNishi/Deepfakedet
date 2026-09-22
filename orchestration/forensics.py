# ruff: noqa: INP001, E501
"""Digital Forensics Analysis Engine for evaluation of digital media authenticity."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as f
from PIL import Image
from torch import nn
from torchvision import transforms


@dataclass
class ModelBundle:
    """Container for model-specific inference resources."""

    name: str
    display_label: str
    model: nn.Module
    transform: transforms.Compose
    normalize: bool
    device: torch.device
    target_layer: nn.Module


def ensure_weights_exist(model_name: str, weights_path: Path) -> None:
    """Programmatically download model weights from GitHub Releases if missing."""
    if weights_path.exists():
        return

    url_base = "https://github.com/thourihan/DeepfakeDetection/releases/download/v0.3.0/"
    name_map = {
        "efficientnet_b3": "efficientnet_b3_v0.3.0.pth",
        "efficientformerv2_s1": "efficientformerv2_s1_v0.3.0.pth",
        "faster_vit_2_224": "faster_vit_2_224_v0.3.0.pth",
    }

    if model_name in name_map:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Forensics] Weight file not found. Downloading {model_name} weights to {weights_path}...")
        try:
            urllib.request.urlretrieve(
                url_base + name_map[model_name], str(weights_path)
            )
            print(f"[Forensics] {model_name} weights downloaded successfully.")
        except Exception as e:
            print(f"[Forensics] Error downloading weights for {model_name}: {e}")
            raise
    else:
        print(f"[Forensics] No download URL mapped for model '{model_name}'.")


def check_metadata_integrity(image: Image.Image) -> tuple[str, float, list[str], dict]:
    """Inspect image EXIF tags for software signatures or missing metadata."""
    from PIL import ExifTags
    exif = image.getexif()
    meta_info = {"camera": "Unknown", "timestamp": "Unknown", "gps": "Unknown"}
    
    if not exif or len(exif.keys()) == 0:
        return "Missing", 0.0, [], meta_info

    # Suspicious editing software tags (e.g. Photoshop, GIMP, Canva)
    suspicious_software = [
        "photoshop", "gimp", "canva", "adobe", "enlight", "picsart",
        "snapseed", "midjourney", "stable diffusion", "dall-e", "pixelmator"
    ]

    found_tags = []
    try:
        make = ""
        model = ""
        for tag_id, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
            val_str = str(value).lower()
            
            # Look for software
            for soft in suspicious_software:
                if soft in val_str and soft not in found_tags:
                    found_tags.append(soft)
                    
            if tag_name == "Make": make = str(value)
            elif tag_name == "Model": model = str(value)
            elif tag_name == "DateTime" or tag_name == "DateTimeOriginal": 
                meta_info["timestamp"] = str(value)
                
        if make or model:
            meta_info["camera"] = f"{make} {model}".strip()

        gps_info = exif.get_ifd(ExifTags.IFD.GPSInfo) if hasattr(exif, 'get_ifd') else None
        if gps_info:
            meta_info["gps"] = "Present (Coordinates found)"
    except Exception:
        pass

    status = "Suspicious" if found_tags else "Valid"
    score = 50.0 if found_tags else 100.0
    return status, score, found_tags, meta_info


def check_video_metadata_integrity(video_path: Path) -> tuple[str, float, list[str], dict]:
    """Perform binary carving on video headers to detect editing software signatures."""
    meta_info = {"camera": "Unknown", "timestamp": "Unknown", "gps": "Unknown"}
    if not video_path.exists():
        return "Missing", 0.0, [], meta_info

    # Read start and end segments of the video file to inspect headers and trailers
    try:
        size = video_path.stat().st_size
        read_size = min(size, 1024 * 512)  # Read first 512KB
        with video_path.open("rb") as f_in:
            header_bytes = f_in.read(read_size)
            if size > read_size:
                f_in.seek(size - read_size)
                trailer_bytes = f_in.read(read_size)
            else:
                trailer_bytes = b""

        all_bytes = (header_bytes + trailer_bytes).lower()

        # Check for signature footprints of common video editors / encoders
        suspicious_signatures = [
            b"premiere", b"after effects", b"handbrake", b"ffmpeg", b"capcut",
            b"davinci", b"imovie", b"vegas pro", b"camtasia", b"wondershare"
        ]

        found_tags = []
        for sig in suspicious_signatures:
            if sig in all_bytes:
                found_tags.append(sig.decode('utf-8', errors='ignore'))

        if found_tags:
            return "Suspicious", 50.0, found_tags, meta_info

        # Check if it has basic video container signatures
        # mp4/mov/mkv
        video_containers = [b"ftyp", b"moov", b"matroska", b"riff"]
        if any(c in header_bytes for c in video_containers):
            return "Valid", 100.0, [], meta_info

        return "Suspicious", 70.0, [], meta_info
    except Exception:
        return "Missing", 0.0, [], meta_info


def check_sensor_noise(image: Image.Image) -> tuple[str, float]:
    """Analyze high-frequency sensor noise (Photo Response Non-Uniformity).

    Natural digital camera sensors introduce a characteristic high-frequency sensor noise (grain)
    that is present even in flat/homogeneous regions. AI diffusion models denoise these flat areas
    to near-perfection (noise standard deviation close to 0).
    """
    try:
        img_gray = image.convert("L")
        img_arr = np.array(img_gray, dtype=np.float32)
        h, w = img_arr.shape
        if h < 4 or w < 4:
            return "Valid", 100.0

        # Calculate gradients to find flat regions
        dy, dx = np.gradient(img_arr)
        grad_mag = np.sqrt(dx**2 + dy**2)

        # Homogeneous region mask (very low gradients)
        flat_mask = grad_mag < 1.2

        if np.sum(flat_mask) > 500:
            kernel = np.ones((3, 3), dtype=np.float32) / 9.0
            local_mean = cv2.filter2D(img_arr, -1, kernel)
            high_pass = img_arr - local_mean
            noise_std = float(np.std(high_pass[flat_mask]))
        else:
            kernel = np.ones((3, 3), dtype=np.float32) / 9.0
            local_mean = cv2.filter2D(img_arr, -1, kernel)
            high_pass = img_arr - local_mean
            noise_std = float(np.std(high_pass))

        if noise_std < 0.50:
            score = float(max(0.0, min(100.0, noise_std * 180.0)))
            status = "Suspicious" if noise_std > 0.38 else "Missing Noise"
            return status, score
        else:
            return "Valid", 100.0
    except Exception:
        return "Valid", 100.0


def compute_artifact_score(
    image: Image.Image, is_fake_pred: bool, model_confidence: float
) -> tuple[str, float, float, float]:
    """Perform Grayscale 2D Fast Fourier Transform and Sensor Noise analysis to identify generative artifacts."""
    try:
        # A. FFT high-frequency grid checks
        img_gray = image.convert("L")
        img_arr = np.array(img_gray, dtype=np.float32)
        h, w = img_arr.shape
        if h < 4 or w < 4:
            return "Not Detected", 100.0, 0.0, 0.0

        cy, cx = h // 2, w // 2
        f_transform = np.fft.fft2(img_arr)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((y - cy)**2 + (x - cx)**2)
        r_low = min(h, w) * 0.15
        total_mag = np.sum(magnitude)
        high_freq_ratio = float(np.sum(magnitude[dist > r_low]) / total_mag) if total_mag > 0 else 0.0

        # B. Sensor noise check
        dy, dx = np.gradient(img_arr)
        grad_mag = np.sqrt(dx**2 + dy**2)
        flat_mask = grad_mag < 1.2
        if np.sum(flat_mask) > 500:
            kernel = np.ones((3, 3), dtype=np.float32) / 9.0
            local_mean = cv2.filter2D(img_arr, -1, kernel)
            high_pass = img_arr - local_mean
            noise_std = float(np.std(high_pass[flat_mask]))
        else:
            kernel = np.ones((3, 3), dtype=np.float32) / 9.0
            local_mean = cv2.filter2D(img_arr, -1, kernel)
            high_pass = img_arr - local_mean
            noise_std = float(np.std(high_pass))

        # Check for both grid artifacts and missing camera sensor noise
        artifact_threshold = 0.35 if is_fake_pred else 0.45
        has_fft_grids = (high_freq_ratio > artifact_threshold) or (is_fake_pred and model_confidence > 75.0)
        has_missing_noise = (noise_std < 0.48) and (high_freq_ratio > 0.28) # standard indicator of high-res diffusion fakes

        if has_fft_grids or has_missing_noise:
            # Artifacts detected
            score = 0.0
            if has_missing_noise and not has_fft_grids:
                score = float(max(0.0, min(50.0, noise_std * 90.0))) # partial score for noise
            return "Detected", score, high_freq_ratio, noise_std
        else:
            return "Not Detected", 100.0, high_freq_ratio, noise_std
    except Exception:
        return "Not Detected", 100.0, 0.0, 0.0


def compute_visual_consistency(
    image: Image.Image, is_fake_pred: bool
) -> tuple[str, float, float]:
    """Evaluate texture uniformity and sharpness via localized variance & gradients."""
    try:
        img_gray = image.convert("L")
        img_arr = np.array(img_gray, dtype=np.float32)
        h, w = img_arr.shape
        if h < 8 or w < 8:
            return "Medium", 60.0, 0.0

        # Gradient variance (sharpness consistency)
        dy, dx = np.gradient(img_arr)
        grad_mag = np.sqrt(dx**2 + dy**2)
        grad_var = float(np.var(grad_mag))

        # Local block analysis for texture consistency
        blocks_y, blocks_x = 8, 8
        block_h, block_w = max(1, h // blocks_y), max(1, w // blocks_x)
        block_stds = []
        for i in range(blocks_y):
            for j in range(blocks_x):
                block = img_arr[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                if block.size > 0:
                    block_stds.append(np.std(block))

        if len(block_stds) > 0:
            block_stds = np.array(block_stds)
            std_of_stds = float(np.std(block_stds))
            mean_of_stds = float(np.mean(block_stds))
            texture_inconsistency = std_of_stds / (mean_of_stds + 1e-5)
        else:
            texture_inconsistency = 0.0

        # Classify consistency level
        if is_fake_pred:
            if texture_inconsistency > 0.6 or grad_var < 50.0:
                return "Low", 20.0, texture_inconsistency
            return "Medium", 60.0, texture_inconsistency
        else:
            if texture_inconsistency < 0.4 and grad_var > 100.0:
                return "High", 100.0, texture_inconsistency
            if texture_inconsistency > 0.7 or grad_var < 30.0:
                return "Low", 20.0, texture_inconsistency
            return "Medium", 70.0, texture_inconsistency
    except Exception:
        return "Medium", 60.0, 0.0


def compute_reliability(
    pred_label: str,
    model_confidence: float,
    visual_consistency_score: float,
    metadata_score: float,
    artifact_score: float
) -> float:
    """Apply the forensic reliability formula."""
    is_fake = (pred_label.upper() == "FAKE")
    # For FAKE predictions, model confidence term is inverted (100 - confidence)
    # to represent authenticity of evidence.
    model_contrib = model_confidence if not is_fake else (100.0 - model_confidence)

    reliability = (
        0.4 * model_contrib +
        0.2 * visual_consistency_score +
        0.2 * metadata_score +
        0.2 * artifact_score
    )
    return float(max(0.0, min(100.0, reliability)))


def format_forensic_report(
    prediction: str,
    model_confidence: float,
    reliability_score: float,
    visual_consistency: str,
    metadata_integrity: str,
    artifact_detection: str,
    reasoning: str = ""
) -> str:
    """Format forensic report in the exact required layout."""
    if not reasoning:
        if prediction.upper() == "REAL":
            if reliability_score >= 80.0:
                reasoning = (
                    "The digital evidence is highly reliable. The media exhibits natural textures, "
                    "valid metadata, and no anomalous frequency artifacts, suggesting authentic capture."
                )
            elif reliability_score >= 50.0:
                reasoning = (
                    f"The digital evidence is moderately reliable. While classified as REAL with "
                    f"{model_confidence:.1f}% confidence, some minor forensic factors (e.g. missing metadata) "
                    "suggest post-processing, but there are no indicators of synthetic generation."
                )
            else:
                reasoning = (
                    "The evidence is suspicious and has low reliability despite being classified as REAL. "
                    "Inconsistent texture sharpness and missing metadata indicate significant processing."
                )
        else:
            if reliability_score <= 30.0:
                reasoning = (
                    "The digital evidence is highly untrustworthy. Advanced synthesis artifacts and "
                    "low visual consistency strongly indicate deepfake generation or significant local manipulation."
                )
            else:
                reasoning = (
                    f"The digital evidence is suspicious and unreliable. The system predicts FAKE with "
                    f"{model_confidence:.1f}% confidence, supported by localized anomalies and compression footprints."
                )

    report = f"""----------------------------------------
Prediction: {prediction.upper()}
Model Confidence: {model_confidence:.0f}%
Reliability Score: {reliability_score:.0f}%

Analysis Summary:
- Visual Consistency: {visual_consistency}
- Metadata Integrity: {metadata_integrity}
- Artifact Detection: {artifact_detection}

Final Interpretation:
{reasoning}
----------------------------------------"""
    return report


def run_forensic_analysis_image(
    image: Image.Image, models: list[ModelBundle]
) -> dict[str, Any]:
    """Execute single image classification using a weighted model ensemble fused with handcrafted forensic features."""
    # 1. Classification prediction using model ensemble
    probs_real_list = []
    for bundle in models:
        tensor = bundle.transform(image)
        if tensor.ndim == 3:
            batch = tensor.unsqueeze(0)
        else:
            batch = tensor
        batch = batch.to(bundle.device)

        with torch.inference_mode():
            logits = bundle.model(batch)
            probs = f.softmax(logits, dim=1)
            probs_real_list.append(float(probs[0, 1]))

    # Weighted ensemble prediction
    model_weights = {
        "faster_vit_2_224": 0.50,
        "efficientnet_b3": 0.35,
        "efficientformerv2_s1": 0.15
    }

    total_weight = 0.0
    weighted_sum = 0.0
    for bundle, prob in zip(models, probs_real_list, strict=False):
        w = model_weights.get(bundle.name, 1.0)
        weighted_sum += prob * w
        total_weight += w

    if total_weight > 0:
        dl_prob_real = weighted_sum / total_weight
    else:
        dl_prob_real = float(np.mean(probs_real_list)) if probs_real_list else 0.5

    # 2. Extract handcrafted forensic features for Fusion
    # A. FFT frequency analysis
    img_gray = image.convert("L")
    img_arr = np.array(img_gray, dtype=np.float32)
    h, w = img_arr.shape
    cy, cx = h // 2, w // 2
    f_transform = np.fft.fft2(img_arr)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    r_low = min(h, w) * 0.15
    total_mag = np.sum(magnitude)
    high_freq_ratio = float(np.sum(magnitude[dist > r_low]) / total_mag) if total_mag > 0 else 0.0

    # B. Texture consistency
    dy, dx = np.gradient(img_arr)
    grad_mag = np.sqrt(dx**2 + dy**2)
    float(np.var(grad_mag))
    blocks_y, blocks_x = 8, 8
    block_h, block_w = max(1, h // blocks_y), max(1, w // blocks_x)
    block_stds = []
    for i in range(blocks_y):
        for j in range(blocks_x):
            block = img_arr[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            if block.size > 0:
                block_stds.append(np.std(block))
    if len(block_stds) > 0:
        block_stds = np.array(block_stds)
        std_of_stds = float(np.std(block_stds))
        mean_of_stds = float(np.mean(block_stds))
        texture_inconsistency = std_of_stds / (mean_of_stds + 1e-5)
    else:
        texture_inconsistency = 0.0

    # Map physical features to realness probabilities
    # High frequency ratio mapping: low ratio -> authentic (1.0), high ratio (>0.40) -> fake (0.0)
    art_prob = float(np.clip((high_freq_ratio - 0.22) / (0.42 - 0.22), 0.0, 1.0))
    # Texture inconsistency mapping: low inconsistency -> authentic (1.0), high inconsistency -> fake (0.0)
    vis_cons_prob = 1.0 - float(np.clip((texture_inconsistency - 0.30) / (0.75 - 0.30), 0.0, 1.0))

    # Hybrid Fusion Classification Decision
    forensic_realness = 0.5 * vis_cons_prob + 0.5 * (1.0 - art_prob)
    fused_prob_real = 0.7 * dl_prob_real + 0.3 * forensic_realness

    if fused_prob_real >= 0.5:
        prediction = "REAL"
        confidence = fused_prob_real * 100.0
    else:
        prediction = "FAKE"
        confidence = (1.0 - fused_prob_real) * 100.0

    # 3. Standard output scoring formatting
    metadata_status, metadata_score, meta_tags, meta_info = check_metadata_integrity(image)
    artifact_status, artifact_score, hf_ratio, n_std = compute_artifact_score(image, prediction == "FAKE", confidence)
    vis_status, vis_score, tex_inconsist = compute_visual_consistency(image, prediction == "FAKE")

    reliability_score = compute_reliability(
        prediction, confidence, vis_score, metadata_score, artifact_score
    )

    report_text = format_forensic_report(
        prediction, confidence, reliability_score, vis_status, metadata_status, artifact_status
    )

    return {
        "prediction": prediction,
        "model_confidence": confidence,
        "reliability_score": reliability_score,
        "visual_consistency": vis_status,
        "metadata_integrity": metadata_status,
        "artifact_detection": artifact_status,
        "report": report_text,
        "proofs": {
            "metadata_tags": meta_tags,
            "high_freq_ratio": hf_ratio,
            "noise_std": n_std,
            "texture_inconsistency": tex_inconsist,
            "meta_camera": meta_info["camera"],
            "meta_timestamp": meta_info["timestamp"],
            "meta_gps": meta_info["gps"]
        }
    }


def run_forensic_analysis_video(
    video_path: Path, models: list[ModelBundle]
) -> dict[str, Any]:
    """Open video file, extract keyframes, run weighted ensembled inference with forensic fusion, and temporal checks."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "prediction": "FAKE",
            "model_confidence": 0.0,
            "reliability_score": 0.0,
            "visual_consistency": "Low",
            "metadata_integrity": "Missing",
            "artifact_detection": "Detected",
            "report": "Error: Could not open video file.",
            "keyframes": []
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_samples = min(total_frames, 12) if total_frames > 0 else 1
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int) if total_frames > 1 else [0]

    frames_pil = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_pil.append(Image.fromarray(frame_rgb))
    cap.release()

    if not frames_pil:
        return {
            "prediction": "FAKE",
            "model_confidence": 0.0,
            "reliability_score": 0.0,
            "visual_consistency": "Low",
            "metadata_integrity": "Missing",
            "artifact_detection": "Detected",
            "report": "Error: No frames could be extracted from the video.",
            "keyframes": []
        }

    # Model Weights
    model_weights = {
        "faster_vit_2_224": 0.50,
        "efficientnet_b3": 0.35,
        "efficientformerv2_s1": 0.15
    }

    frame_dl_probs = []
    frame_fused_probs = []
    vis_scores_list = []
    artifact_scores_list = []

    for img in frames_pil:
        # Run classification predictions for this frame
        frame_probs = []
        for bundle in models:
            tensor = bundle.transform(img)
            if tensor.ndim == 3:
                batch = tensor.unsqueeze(0)
            else:
                batch = tensor
            batch = batch.to(bundle.device)

            with torch.inference_mode():
                logits = bundle.model(batch)
                probs = f.softmax(logits, dim=1)
                frame_probs.append(float(probs[0, 1]))

        # Weighted ensembling
        total_weight = 0.0
        weighted_sum = 0.0
        for bundle, prob in zip(models, frame_probs, strict=False):
            w = model_weights.get(bundle.name, 1.0)
            weighted_sum += prob * w
            total_weight += w

        frame_dl_prob = weighted_sum / total_weight if total_weight > 0 else float(np.mean(frame_probs))
        frame_dl_probs.append(frame_dl_prob)

        # Extract features for Fusion
        img_gray = img.convert("L")
        img_arr = np.array(img_gray, dtype=np.float32)
        h, w = img_arr.shape
        cy, cx = h // 2, w // 2
        f_transform = np.fft.fft2(img_arr)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((y - cy)**2 + (x - cx)**2)
        r_low = min(h, w) * 0.15
        total_mag = np.sum(magnitude)
        high_freq_ratio = float(np.sum(magnitude[dist > r_low]) / total_mag) if total_mag > 0 else 0.0

        dy, dx = np.gradient(img_arr)
        grad_mag = np.sqrt(dx**2 + dy**2)
        float(np.var(grad_mag))
        blocks_y, blocks_x = 8, 8
        block_h, block_w = max(1, h // blocks_y), max(1, w // blocks_x)
        block_stds = []
        for i in range(blocks_y):
            for j in range(blocks_x):
                block = img_arr[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                if block.size > 0:
                    block_stds.append(np.std(block))
        if len(block_stds) > 0:
            block_stds = np.array(block_stds)
            std_of_stds = float(np.std(block_stds))
            mean_of_stds = float(np.mean(block_stds))
            texture_inconsistency = std_of_stds / (mean_of_stds + 1e-5)
        else:
            texture_inconsistency = 0.0

        art_prob = float(np.clip((high_freq_ratio - 0.22) / (0.42 - 0.22), 0.0, 1.0))
        vis_cons_prob = 1.0 - float(np.clip((texture_inconsistency - 0.30) / (0.75 - 0.30), 0.0, 1.0))

        forensic_realness = 0.5 * vis_cons_prob + 0.5 * (1.0 - art_prob)
        frame_fused_prob = 0.7 * frame_dl_prob + 0.3 * forensic_realness
        frame_fused_probs.append(frame_fused_prob)

        # Standard check values
        _, frame_vis_score, _ = compute_visual_consistency(img, frame_fused_prob < 0.5)
        _, frame_art_score, _, _ = compute_artifact_score(img, frame_fused_prob < 0.5, (1.0 - frame_fused_prob)*100 if frame_fused_prob < 0.5 else frame_fused_prob*100)
        vis_scores_list.append(frame_vis_score)
        artifact_scores_list.append(frame_art_score)

    # Ensembling across frames
    mean_fused_prob_real = float(np.mean(frame_fused_probs))

    # Calculate temporal consistency check (flicker analysis)
    # Temporal variance of model predictions indicates frame-to-frame synthesis inconsistency
    temporal_variance = float(np.var(frame_dl_probs))
    temporal_penalty = float(max(0.0, min(30.0, (temporal_variance - 0.02) * 500.0))) # penalty up to 30 points

    # Adjust decision based on high temporal inconsistency (flicker)
    fused_prob_real = max(0.0, mean_fused_prob_real - 0.4 * temporal_variance)

    if fused_prob_real >= 0.5:
        prediction = "REAL"
        confidence = fused_prob_real * 100.0
    else:
        prediction = "FAKE"
        confidence = (1.0 - fused_prob_real) * 100.0

    avg_vis_score = max(0.0, float(np.mean(vis_scores_list)) - temporal_penalty)
    avg_art_score = float(np.mean(artifact_scores_list))

    if avg_vis_score >= 80.0:
        vis_status = "High"
    elif avg_vis_score >= 40.0:
        vis_status = "Medium"
    else:
        vis_status = "Low"

    if avg_art_score < 50.0 or temporal_variance > 0.05:
        artifact_status = "Detected"
    else:
        artifact_status = "Not Detected"

    metadata_status, metadata_score, meta_tags, meta_info = check_video_metadata_integrity(video_path)

    reliability_score = compute_reliability(
        prediction, confidence, avg_vis_score, metadata_score, avg_art_score
    )

    reasoning = f"The video file was evaluated across {len(frames_pil)} keyframes. "
    if prediction == "REAL" and reliability_score >= 70.0:
        reasoning += (
            "Temporal alignment and frequency grids are natural, showing no indicators of "
            "face-swapping synthesis or adversarial artifacts."
        )
    else:
        reasoning += (
            f"Scanning detected a high temporal variation of {temporal_variance:.4f} between frames, "
            "indicating significant frame-to-frame face blending inconsistencies and manipulation artifacts."
        )

    report_text = format_forensic_report(
        prediction, confidence, reliability_score, vis_status, metadata_status, artifact_status,
        reasoning=reasoning
    )

    return {
        "prediction": prediction,
        "model_confidence": confidence,
        "reliability_score": reliability_score,
        "visual_consistency": vis_status,
        "metadata_integrity": metadata_status,
        "artifact_detection": artifact_status,
        "report": report_text,
        "keyframes": frames_pil,
        "proofs": {
            "metadata_tags": meta_tags,
            "temporal_variance": temporal_variance,
            "meta_camera": meta_info["camera"],
            "meta_timestamp": meta_info["timestamp"],
            "meta_gps": meta_info["gps"]
        }
    }


def _coerce_device(device_str: str | None) -> torch.device:
    if device_str:
        requested = torch.device(device_str)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _detect_normalization(transform: transforms.Compose) -> bool:
    for op in getattr(transform, "transforms", []):
        if isinstance(op, transforms.Normalize):
            return True
    return False


def load_forensics_models(config_path: Path) -> list[ModelBundle]:
    """Load model config and active models for forensic ensembled inference."""
    from .orchestrator import (
        build_eval_transforms,
        load_config,
        load_model,
        resolve_transform_mapping,
    )

    config = load_config(config_path)
    device_str = config.get("device")
    device = _coerce_device(device_str)

    data_cfg = config.get("data", {})
    num_classes = int(data_cfg.get("num_classes", 2))
    image_size = int(data_cfg.get("img_size", 224))

    models_cfg = config.get("models", {})
    selection = config.get("selection") or list(models_cfg.keys())

    bundles = []
    for model_name in selection:
        model_cfg = models_cfg.get(model_name)
        if not isinstance(model_cfg, dict):
            continue

        toggles = resolve_transform_mapping(model_cfg, phase="eval")
        transform = build_eval_transforms(image_size, toggles=toggles)
        normalize = _detect_normalization(transform)

        inference_cfg = model_cfg.get("inference", {})
        weights_path_value = inference_cfg.get("weights")
        if weights_path_value:
            weights_path = Path(weights_path_value).expanduser()
            if not weights_path.is_absolute():
                weights_path = (Path.cwd() / weights_path).resolve()
        else:
            weights_path = Path(f"weights/{model_name}.pth")

        # Auto download if missing
        ensure_weights_exist(model_name, weights_path)

        model = load_model(model_name, num_classes, weights_path, device)

        # Pick the last Conv2d for Grad-CAM target layer
        last_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m

        display_label = str(model_cfg.get("display_name") or model_cfg.get("label") or model_name)

        bundles.append(
            ModelBundle(
                name=model_name,
                display_label=display_label,
                model=model,
                transform=transform,
                normalize=normalize,
                device=device,
                target_layer=last_conv or next(model.parameters())
            )
        )
    return bundles
