"""Verify the fixed-resolution source chart directly from its paper PDF."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from development.loss_model_analysis.pdf_sources import chart_image, chart_pixels_sha256


if __name__ == "__main__":
    image = chart_image("ko_hub_mach")
    print(f"Verified Figure 6 in memory: {image.size}; pixels {chart_pixels_sha256('ko_hub_mach')}")
