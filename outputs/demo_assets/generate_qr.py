"""
generate_qr.py

Generates high-resolution, branded QR code linking directly to the live
Hugging Face Spaces demonstration for Func2OCT.
Saves the artifact into outputs/demo_assets/hf_spaces_demo_qr.png.
"""

from __future__ import annotations

import os
from pathlib import Path
from PIL import Image, ImageDraw
import qrcode

HF_SPACES_URL = "https://huggingface.co/spaces/Charan9441/Fund2Oct"


def create_branded_qr(
    target_url: str = HF_SPACES_URL,
    output_path: Optional[Path] = None,
) -> Path:
    """Creates a clinical/research branded QR code banner linking to HF Spaces."""
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "hf_spaces_demo_qr.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Generate core QR Code
    qr = qrcode.QRCode(
        version=3,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=12,
        border=3,
    )
    qr.add_data(target_url)
    qr.make(fit=True)

    qr_img = qr.make_image(
        fill_color="#00695c",  # Deep Teal
        back_color="#ffffff",
    ).convert("RGBA")

    qr_w, qr_h = qr_img.size

    # 2. Build Branded Presentation Canvas
    card_w = qr_w + 120
    card_h = qr_h + 260
    canvas = Image.new("RGBA", (card_w, card_h), "#ffffff")
    draw = ImageDraw.Draw(canvas)

    # Decorative header background
    draw.rectangle([0, 0, card_w, 110], fill="#004d40")

    # Header Title Text
    draw.text((card_w // 2, 35), "Func2OCT Web Studio", fill="#ffffff", anchor="mm")
    draw.text((card_w // 2, 65), "Hugging Face Spaces Live Demonstration", fill="#80cbc4", anchor="mm")
    draw.text((card_w // 2, 90), "Clinical Auto-Capture & Latent Diffusion Synthesis", fill="#e0f2f1", anchor="mm")

    # Paste QR Code in center
    qr_x = (card_w - qr_w) // 2
    qr_y = 130
    canvas.paste(qr_img, (qr_x, qr_y), qr_img)

    # Decorative border around QR code
    draw.rectangle(
        [qr_x - 4, qr_y - 4, qr_x + qr_w + 4, qr_y + qr_h + 4],
        outline="#00897b",
        width=3,
    )

    # Regulatory & Access Footer
    foot_y = qr_y + qr_h + 20
    draw.text((card_w // 2, foot_y), "SCAN WITH SMARTPHONE CAMERA TO LAUNCH", fill="#004d40", anchor="mm")
    draw.text((card_w // 2, foot_y + 24), target_url, fill="#00796b", anchor="mm")

    # CDSCO SaMD Disclaimer pill badge
    badge_rect = [30, foot_y + 45, card_w - 30, foot_y + 90]
    draw.rectangle(badge_rect, fill="#fff8e1", outline="#ffe082", width=1)
    draw.text(
        (card_w // 2, foot_y + 60),
        "CDSCO Rule 51 (MDR 2017) | Investigational SaMD",
        fill="#b71c1c",
        anchor="mm",
    )
    draw.text(
        (card_w // 2, foot_y + 78),
        "For Academic Research & Clinical Feasibility Benchmarking Only",
        fill="#5d4037",
        anchor="mm",
    )

    # Save artifact
    canvas_rgb = canvas.convert("RGB")
    canvas_rgb.save(str(output_path), quality=95)
    print(f"[QR Code] Successfully generated branded QR code at: {output_path}")
    return output_path


if __name__ == "__main__":
    create_branded_qr()
