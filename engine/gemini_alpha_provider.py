"""
Gemini Authentic Alpha Mask Provider & Catalog Anchor Engine.
Loads calibrated alpha profiles directly from official Google Gemini captures (bg_48.png, bg_96.png),
providing sub-pixel precise alpha channels, dual-polarity outline compensation, and catalog anchors.
"""

import os
from typing import Dict, Any, List
import numpy as np
from PIL import Image

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

# Cache loaded templates
_ALPHA_CACHE: Dict[int, np.ndarray] = {}

def load_official_alpha_map(size: int = 48) -> np.ndarray:
    """
    Loads official calibrated Gemini watermark alpha map.
    Returns float32 numpy array with values in [0.0, ~0.51].
    """
    if size <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    if size in _ALPHA_CACHE:
        return _ALPHA_CACHE[size].copy()

    filename = f"bg_{size}.png" if size in [48, 96] else "bg_48.png"
    filepath = os.path.join(ASSETS_DIR, filename)

    if os.path.exists(filepath):
        im = Image.open(filepath)
        arr = np.array(im).astype(np.float32)
        if arr.ndim == 3:
            alpha = np.max(arr[:, :, :3], axis=-1) / 255.0
        else:
            alpha = arr / 255.0

        if size not in [48, 96] or alpha.shape[0] != size:
            im_alpha = Image.fromarray((alpha * 255.0).astype(np.uint8))
            im_resized = im_alpha.resize((size, size), Image.Resampling.LANCZOS)
            alpha = np.array(im_resized).astype(np.float32) / 255.0
    else:
        alpha = _generate_fallback_gemini_profile(size)

    _ALPHA_CACHE[size] = alpha
    return alpha.copy()


def get_official_gemini_sparkle(
    size: int = 48,
    alpha_gain: float = 1.0,
    dark_polarity: bool = False
) -> np.ndarray:
    """
    Returns the official Google Gemini sparkle alpha profile scaled to requested size and gain.
    """
    if size <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    base_alpha = load_official_alpha_map(size)
    scaled = np.clip(base_alpha * alpha_gain, 0.0, 0.99).astype(np.float32)
    return scaled


def _generate_fallback_gemini_profile(size: int) -> np.ndarray:
    """
    Procedural fallback closely matching the organic curvature of Gemini sparkle logo.
    """
    if size <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    alpha = np.zeros((size, size), dtype=np.float32)
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    radius = size / 2.0

    for y in range(size):
        for x in range(size):
            dx = abs(x - cx) / radius
            dy = abs(y - cy) / radius
            r = np.sqrt(dx * dx + dy * dy)
            p = 0.52
            d_ast = (dx ** p + dy ** p) ** (1.0 / p)
            if d_ast <= 1.0:
                flare = (1.0 - d_ast) ** 1.6
                core = np.exp(-5.5 * r * r)
                val = 0.52 * flare + 0.48 * core
                alpha[y, x] = min(0.505, max(0.0, val * 0.505))
    return alpha


def get_catalog_watermark_anchors(width: int, height: int) -> List[Dict[str, Any]]:
    """
    Returns prioritized list of candidate watermark positions based on Google Gemini's
    discrete resolution tier catalog and margin specifications.
    
    Each candidate is a dict:
        {'x': int, 'y': int, 'size': int, 'margin_x': int, 'margin_y': int, 'corner': str, 'name': str}
    """
    if width <= 0 or height <= 0:
        return []

    anchors: List[Dict[str, Any]] = []

    def add_anchor(s: int, mx: int, my: int, corner: str, name: str) -> None:
        if corner == 'br':
            ax = width - mx - s
            ay = height - my - s
        elif corner == 'bl':
            ax = mx
            ay = height - my - s
        else:
            return

        if 0 <= ax <= width - s and 0 <= ay <= height - s:
            if not any(a['x'] == ax and a['y'] == ay and a['size'] == s for a in anchors):
                anchors.append({
                    'size': s, 'margin_x': mx, 'margin_y': my,
                    'x': ax, 'y': ay,
                    'corner': corner, 'name': name
                })

    # 1. Primary Bottom-Right Imagen 3 & Gemini Standards (Checked for ALL images)
    # 48px @ margin 96px (Current Imagen 3 standard)
    add_anchor(48, 96, 96, 'br', '1k-large-margin (Current)')
    # 48px @ margin 32px (Classic Gemini standard - e.g. 20260520-1.png)
    add_anchor(48, 32, 32, 'br', '1k-small-margin (Classic)')
    # 48px @ margin 64px / 48px / 89px / 73px
    add_anchor(48, 64, 64, 'br', 'br-48px-m64')
    add_anchor(48, 48, 48, 'br', 'br-48px-m48')
    add_anchor(48, 89, 89, 'br', 'br-48px-m89')
    add_anchor(48, 73, 73, 'br', 'br-48px-m73')

    # 2. Large 2K/4K Resolution Standards
    if width > 1024 or height > 1024:
        add_anchor(96, 192, 192, 'br', '2k-new-margin (Current)')
        add_anchor(96, 64, 64, 'br', '2k-legacy-margin')
        add_anchor(96, 96, 96, 'br', '2k-mid-margin')

    # 3. Small v2 Variants
    add_anchor(36, 96, 96, 'br', 'br-36px-m96')
    add_anchor(36, 32, 32, 'br', 'br-36px-m32')
    add_anchor(42, 82, 82, 'br', 'br-42px-m82')

    # 4. Bottom-Left Standards (RTL mode / localized Gemini exports)
    add_anchor(48, 96, 96, 'bl', 'bl-48px-m96')
    add_anchor(48, 32, 32, 'bl', 'bl-48px-m32')

    return anchors
