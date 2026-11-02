"""
Synthetic Data Engine & Sparkle Generator
Synthesizes thousands of photo-realistic training pairs (Watermarked, Clean, Mask)
simulating real-world Google Gemini & Imagen 3 watermarking behavior.
"""

import numpy as np
import io
import math
import os
from PIL import Image

def generate_procedural_sparkle(side=48):
    """
    Generates exact 4-pointed Gemini sparkle alpha map matching Google's mathematical formulation:
    Outer flares along axes (x=0, y=0) with smooth hyperbolic decay + circular core.
    """
    cx = (side - 1) / 2.0
    cy = (side - 1) / 2.0
    r_max = side / 2.0
    alpha = np.zeros((side, side), dtype=np.float32)

    for y in range(side):
        dy = abs(y - cy) / r_max
        for x in range(side):
            dx = abs(x - cx) / r_max
            r = math.sqrt(dx * dx + dy * dy)
            if r > 1.0:
                continue

            # 4-pointed star curve: dx^p + dy^p <= 1 with p ~= 0.5
            p = 0.55
            star_dist = (dx ** p + dy ** p) ** (1.0 / p)
            if star_dist <= 1.0:
                # Radial and astroid falloff
                flare = (1.0 - star_dist) ** 1.8
                # Central core glow
                core = math.exp(-6.0 * r * r)
                val = 0.55 * flare + 0.45 * core
                alpha[y, x] = min(1.0, max(0.0, val))

    # Normalize max opacity to ~0.50 (Gemini default)
    max_v = np.max(alpha)
    if max_v > 0:
        alpha = (alpha / max_v) * 0.52
    return alpha


class SyntheticWatermarkDataset:
    """
    Generates realistic training batches on the fly.
    """
    def __init__(self, patch_size=128):
        self.patch_size = patch_size
        self.cached_sparkles = {
            s: generate_procedural_sparkle(s)
            for s in [32, 36, 40, 48, 56, 64, 80, 96]
        }

    def generate_background_patch(self):
        """Generates diverse background texture (gradients, textures, edges)."""
        S = self.patch_size
        mode = np.random.choice(['gradient', 'solid_noise', 'edge', 'multi_color'])

        if mode == 'gradient':
            c1 = np.random.uniform(0.1, 0.9, size=(3, 1, 1))
            c2 = np.random.uniform(0.1, 0.9, size=(3, 1, 1))
            grad_x = np.linspace(0, 1, S).reshape(1, 1, S)
            grad_y = np.linspace(0, 1, S).reshape(1, S, 1)
            img = c1 * grad_x + c2 * grad_y
        elif mode == 'solid_noise':
            base_col = np.random.uniform(0.1, 0.9, size=(3, 1, 1))
            noise = np.random.normal(0, 0.04, size=(3, S, S))
            img = base_col + noise
        elif mode == 'edge':
            split = np.random.randint(S // 4, 3 * S // 4)
            c1 = np.random.uniform(0.1, 0.9, size=(3, 1, 1))
            c2 = np.random.uniform(0.1, 0.9, size=(3, 1, 1))
            img = np.tile(c1, (1, S, split))
            img2 = np.tile(c2, (1, S, S - split))
            img = np.concatenate([img, img2], axis=2)
        else:
            img = np.random.uniform(0.2, 0.8, size=(3, S, S))

        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def generate_sample(self):
        """
        Creates a single (watermarked, clean, mask, bbox) training sample.
        """
        S = self.patch_size
        clean = self.generate_background_patch()

        # Pick random sparkle size
        s_size = int(np.random.choice([32, 36, 40, 48, 56, 64]))
        sparkle = self.cached_sparkles.get(s_size, generate_procedural_sparkle(s_size))

        # Random opacity scale
        gain = np.random.uniform(0.6, 1.4)
        sp_alpha = np.clip(sparkle * gain, 0.0, 0.95)

        # Random location
        margin = np.random.randint(6, 24)
        corner = np.random.choice(['br', 'bl', 'tr', 'tl', 'center'])

        if corner == 'br':
            x0 = S - margin - s_size
            y0 = S - margin - s_size
        elif corner == 'bl':
            x0 = margin
            y0 = S - margin - s_size
        elif corner == 'tr':
            x0 = S - margin - s_size
            y0 = margin
        elif corner == 'tl':
            x0 = margin
            y0 = margin
        else:
            x0 = np.random.randint(0, S - s_size)
            y0 = np.random.randint(0, S - s_size)

        x0 = max(0, min(S - s_size, x0))
        y0 = max(0, min(S - s_size, y0))

        # Composite watermark onto clean image:
        # Watermark in Gemini is white star: I_wm = I_clean * (1 - alpha) + 1.0 * alpha
        watermarked = clean.copy()
        mask = np.zeros((1, S, S), dtype=np.float32)

        alpha_expanded = sp_alpha[np.newaxis, :, :] # (1, s, s)
        patch_clean = clean[:, y0:y0+s_size, x0:x0+s_size]

        # In linear light / sRGB
        watermarked[:, y0:y0+s_size, x0:x0+s_size] = patch_clean * (1.0 - alpha_expanded) + 1.0 * alpha_expanded
        mask[0, y0:y0+s_size, x0:x0+s_size] = np.where(sp_alpha > 0.02, 1.0, 0.0)

        # Add simulated JPEG compression noise
        if np.random.rand() > 0.3:
            noise = np.random.normal(0, 0.015, size=watermarked.shape)
            watermarked = np.clip(watermarked + noise, 0.0, 1.0).astype(np.float32)

        bbox = np.array([
            (x0 + s_size / 2.0) / S,
            (y0 + s_size / 2.0) / S,
            s_size / float(S),
            s_size / float(S)
        ], dtype=np.float32)

        return watermarked, clean, mask, bbox

    def get_batch(self, batch_size=4):
        """Returns batched numpy tensors."""
        wm_list, clean_list, mask_list, box_list = [], [], [], []
        for _ in range(batch_size):
            w, c, m, b = self.generate_sample()
            wm_list.append(w)
            clean_list.append(c)
            mask_list.append(m)
            box_list.append(b)
        return (
            np.stack(wm_list, axis=0),
            np.stack(clean_list, axis=0),
            np.stack(mask_list, axis=0),
            np.stack(box_list, axis=0)
        )
