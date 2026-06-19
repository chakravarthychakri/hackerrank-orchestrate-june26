from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat, ImageFilter

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def assess_image(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "readable": False,
        "width": None,
        "height": None,
        "brightness": None,
        "sharpness_proxy": None,
        "flags": [],
    }
    if not path.exists():
        info["flags"].append("damage_not_visible")
        return info
    if path.suffix.lower() not in ALLOWED_EXTS:
        info["flags"].append("non_original_image")
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            info["readable"] = True
            info["width"] = w
            info["height"] = h
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            brightness = float(stat.mean[0])
            # A cheap no-op dependency-free sharpness proxy: edge variance after FIND_EDGES.
            edges = gray.filter(ImageFilter.FIND_EDGES)
            sharpness = float(ImageStat.Stat(edges).var[0])
            info["brightness"] = round(brightness, 2)
            info["sharpness_proxy"] = round(sharpness, 2)
            if w < 320 or h < 240:
                info["flags"].append("cropped_or_obstructed")
            if brightness < 35 or brightness > 235:
                info["flags"].append("low_light_or_glare")
            if sharpness < 30:
                info["flags"].append("blurry_image")
    except Exception:
        info["flags"].append("non_original_image")
    return info


def encode_image_for_gemini(path: Path, max_side: int = 1600, jpeg_quality: int = 85) -> dict[str, str]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        data = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"mime_type": "image/jpeg", "data": data}
