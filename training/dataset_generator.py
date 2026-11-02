"""
Synthetic Data Engine & Authentic Gemini Sparkle Generator
Synthesizes photo-realistic training pairs (Watermarked, Clean, Mask)
using official Google Gemini & Imagen 3 calibrated alpha profiles.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.gemini_alpha_provider import get_official_gemini_sparkle

def generate_procedural_sparkle(side: int = 48) -> np.ndarray:
    """
    Returns authentic Google Gemini sparkle alpha profile.
    Maintains backward compatibility while using the exact official shape.
    """
    return get_official_gemini_sparkle(side)


class SyntheticWatermarkDataset:
    """
    Generates realistic training batches using authentic Gemini watermark templates.
    """
    def __init__(self, patch_size: int = 128) -> None:
        self.patch_size = patch_size
        self.canonical_sizes = [36, 42, 48, 64, 96]
        self.cached_sparkles = {
            s: get_official_gemini_sparkle(s)
            for s in self.canonical_sizes
        }

    def generate_background_patch(self):
        """Generates diverse background texture (gradients, textures, edges)."""
        S = self.patch_size
        mode = np.random.choice(['gradient', 'solid_noise', 'edge', 'multi_color', 'bright_natural'])

        if mode == 'gradient':
            c1 = np.random.uniform(0.2, 0.9, size=(3, 1, 1))
            c2 = np.random.uniform(0.2, 0.9, size=(3, 1, 1))
            grad_x = np.linspace(0, 1, S).reshape(1, 1, S)
            grad_y = np.linspace(0, 1, S).reshape(1, S, 1)
            img = c1 * grad_x + c2 * grad_y
        elif mode == 'solid_noise':
            base_col = np.random.uniform(0.3, 0.85, size=(3, 1, 1))
            noise = np.random.normal(0, 0.03, size=(3, S, S))
            img = base_col + noise
        elif mode == 'edge':
            split = np.random.randint(S // 4, 3 * S // 4)
            c1 = np.random.uniform(0.2, 0.8, size=(3, 1, 1))
            c2 = np.random.uniform(0.3, 0.9, size=(3, 1, 1))
            img = np.tile(c1, (1, S, split))
            img2 = np.tile(c2, (1, S, S - split))
            img = np.concatenate([img, img2], axis=2)
        elif mode == 'bright_natural':
            # Simulating sky, foliage, portrait tones
            sky = np.array([0.45, 0.65, 0.90]).reshape(3, 1, 1)
            noise = np.random.normal(0, 0.02, size=(3, S, S))
            img = sky + noise
        else:
            img = np.random.uniform(0.3, 0.8, size=(3, S, S))

        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def generate_sample(self):
        """
        Creates a single (watermarked, clean, mask, bbox) training sample
        using authentic Gemini alpha maps.
        """
        S = self.patch_size
        clean = self.generate_background_patch()

        # Pick authentic sparkle size that fits within patch_size S
        valid_sizes = [s for s in self.canonical_sizes if s <= S]
        if not valid_sizes:
            valid_sizes = [min(S, 36)]
        s_size = int(np.random.choice(valid_sizes))
        sparkle = self.cached_sparkles.get(s_size)
        if sparkle is None:
            sparkle = get_official_gemini_sparkle(s_size)

        # Realistic Gemini alpha gain (0.7 to 1.1)
        gain = float(np.random.uniform(0.75, 1.05))
        sp_alpha = np.clip(sparkle * gain, 0.0, 0.95)

        # Bottom-Right prioritized margins
        margin = int(np.random.choice([16, 24, 32, 48]))
        corner = np.random.choice(['br', 'br', 'br', 'bl']) # 75% bottom-right

        if corner == 'br':
            x0 = S - margin - s_size
            y0 = S - margin - s_size
        else:
            x0 = margin
            y0 = S - margin - s_size

        x0 = max(0, min(S - s_size, x0))
        y0 = max(0, min(S - s_size, y0))

        # Composite watermark onto clean image:
        # Authentic linear optical alpha blending: I_wm = I_clean * (1 - alpha) + 1.0 * alpha
        watermarked = clean.copy()
        mask = np.zeros((1, S, S), dtype=np.float32)

        alpha_expanded = sp_alpha[np.newaxis, :, :] # (1, s, s)
        patch_clean = clean[:, y0:y0+s_size, x0:x0+s_size]

        watermarked[:, y0:y0+s_size, x0:x0+s_size] = patch_clean * (1.0 - alpha_expanded) + 1.0 * alpha_expanded
        mask[0, y0:y0+s_size, x0:x0+s_size] = np.where(sp_alpha > 0.02, 1.0, 0.0)

        # Add simulated JPEG compression noise
        if np.random.rand() > 0.3:
            noise = np.random.normal(0, 0.012, size=watermarked.shape)
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
