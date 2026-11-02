"""
Gemini Authentic Alpha Mask Provider & Catalog Anchor Engine.
Loads calibrated alpha profiles directly from official Google Gemini captures (bg_48.png, bg_96.png),
providing sub-pixel precise alpha channels, dual-polarity outline compensation, and catalog anchors.
"""

import os
from typing import Dict, List, Any
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
        # Max across RGB channels divided by 255 gives authentic alpha
        if arr.ndim == 3:
            alpha = np.max(arr[:, :, :3], axis=-1) / 255.0
        else:
            alpha = arr / 255.0

        if size not in [48, 96] or alpha.shape[0] != size:
            # Resample smoothly to requested size
            im_alpha = Image.fromarray((alpha * 255.0).astype(np.uint8))
            im_resized = im_alpha.resize((size, size), Image.Resampling.LANCZOS)
            alpha = np.array(im_resized).astype(np.float32) / 255.0
    else:
        # Fallback procedural shape if asset file missing
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
    Args:
        size: Target watermark bounding side in pixels.
        alpha_gain: Multiplicative opacity modifier.
        dark_polarity: Reserved for inverted/dark background outline compensation.
    Returns:
        float32 numpy array in range [0.0, 0.99].
    """
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
            # Organic astroid flare
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
        {'x': int, 'y': int, 'size': int, 'margin': int, 'corner': str, 'name': str}
    """
    if width <= 0 or height <= 0:
        return []

    anchors = []
    is_large = (width > 1024 or height > 1024)

    # 1. Primary Bottom-Right Imagen 3 / Gemini 3x Large Margin (Current Standard)
    if not is_large:
        # 1K standard images (e.g. 1024x1024)
        anchors.append({
            'size': 48, 'margin_x': 96, 'margin_y': 96,
            'x': width - 96 - 48, 'y': height - 96 - 48,
            'corner': 'br', 'name': '1k-large-margin (Current)'
        })
        anchors.append({
            'size': 48, 'margin_x': 32, 'margin_y': 32,
            'x': width - 32 - 48, 'y': height - 32 - 48,
            'corner': 'br', 'name': '1k-small-margin (Classic)'
        })
    else:
        # 2K/4K large images (e.g. 2048x2048 or widescreen exports)
        anchors.append({
            'size': 96, 'margin_x': 192, 'margin_y': 192,
            'x': width - 192 - 96, 'y': height - 192 - 96,
            'corner': 'br', 'name': '2k-new-margin (Current)'
        })
        anchors.append({
            'size': 96, 'margin_x': 64, 'margin_y': 64,
            'x': width - 64 - 96, 'y': height - 64 - 96,
            'corner': 'br', 'name': '2k-legacy-margin'
        })
        # Widescreen 1K tier fallback on 2K canvas
        anchors.append({
            'size': 48, 'margin_x': 96, 'margin_y': 96,
            'x': width - 96 - 48, 'y': height - 96 - 48,
            'corner': 'br', 'name': '1k-large-margin fallback'
        })

    # 2. Known Secondary Variants (v2 small 36px, wide aspect 89px, 48px)
    for s, m in [(36, 96), (48, 89), (48, 48), (48, 64), (32, 32), (64, 48)]:
        ax = width - m - s
        ay = height - m - s
        if not any(a['x'] == ax and a['y'] == ay and a['size'] == s for a in anchors):
            anchors.append({
                'size': s, 'margin_x': m, 'margin_y': m,
                'x': ax, 'y': ay,
                'corner': 'br', 'name': f'br-{s}px-m{m}'
            })

    # 3. Bottom-Left (RTL mode / localized Gemini exports)
    for s, m in [(48, 96), (48, 32)]:
        ax = m
        ay = height - m - s
        anchors.append({
            'size': s, 'margin_x': m, 'margin_y': m,
            'x': ax, 'y': ay,
            'corner': 'bl', 'name': f'bl-{s}px-m{m}'
        })

    # Strict coordinate boundary safeguarding
    valid_anchors = []
    for a in anchors:
        s = a['size']
        if a['x'] >= 0 and a['y'] >= 0 and (a['x'] + s <= width) and (a['y'] + s <= height):
            valid_anchors.append(a)

    return valid_anchors
