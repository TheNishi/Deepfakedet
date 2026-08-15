"""Thin inference entrypoint that delegates to the orchestrator or runs single-file forensics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from orchestration.forensics import (
    load_forensics_models,
    run_forensic_analysis_image,
    run_forensic_analysis_video,
)
from orchestration.orchestrator import orchestrate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deepfake inference or single-file forensic analysis")
    parser.add_argument("--config", type=Path, default=Path("config/inference.yaml"))
    parser.add_argument("--input", type=Path, help="Path to an input image or video file for forensic analysis")
    args = parser.parse_args()

    if args.input:
        input_path = args.input.resolve()
        if not input_path.exists():
            print(f"Error: Input path '{input_path}' does not exist.", file=sys.stderr)
            sys.exit(1)

        print(f"Analyzing digital evidence: {input_path.name}...")

        # Load active models
        try:
            models = load_forensics_models(args.config.resolve())
        except Exception as e:
            print(f"Error loading models: {e}", file=sys.stderr)
            sys.exit(1)

        # Check file extension
        suffix = input_path.suffix.lower()
        image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

        if suffix in image_suffixes:
            try:
                with Image.open(input_path) as img:
                    img_rgb = img.convert("RGB")
                    result = run_forensic_analysis_image(img_rgb, models)
                    print(result["report"])
            except Exception as e:
                print(f"Error analyzing image: {e}", file=sys.stderr)
                sys.exit(1)
        elif suffix in video_suffixes:
            try:
                result = run_forensic_analysis_video(input_path, models)
                print(result["report"])
            except Exception as e:
                print(f"Error analyzing video: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Error: Unsupported file format '{suffix}'. Supported formats: image ({', '.join(image_suffixes)}) or video ({', '.join(video_suffixes)})", file=sys.stderr)
            sys.exit(1)
    else:
        orchestrate(args.config.resolve(), mode="inference")


if __name__ == "__main__":
    main()

