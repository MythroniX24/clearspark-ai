"""
ClearSpark AI — 100% Real Photography Dataset Loader (20,000+ Images)
Permanently configured to use ONLY real-world photography (COCO, Places365, Unsplash, Flickr).
Synthetic/procedural textures are strictly prohibited per permanent system directive.
"""

import os
import math
import random
import glob
import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def generate_calibrated_gemini_sparkle(size: int) -> np.ndarray:
    """Authentic Google Gemini 4-pointed sparkle alpha channel."""
    alpha = np.zeros((size, size), dtype=np.float32)
    cx = (size - 1.0) / 2.0
    cy = (size - 1.0) / 2.0
    radius = size / 2.0

    p = 0.52
    for y in range(size):
        dy = abs(y - cy) / radius
        for x in range(size):
            dx = abs(x - cx) / radius
            r = math.sqrt(dx * dx + dy * dy)
            d_ast = (dx ** p + dy ** p) ** (1.0 / p)
            if d_ast <= 1.0:
                flare = (1.0 - d_ast) ** 1.6
                core = math.exp(-5.5 * r * r)
                val = 0.52 * flare + 0.48 * core
                alpha[y, x] = min(0.505, max(0.0, val * 0.505))

    return alpha


class RealPhotoWatermarkDataset(Dataset if HAS_TORCH else object):
    """
    100% Real Photography Dataset Loader.
    Loads real images from COCO 2017, Places365, Unsplash, or local directories.
    Applies authentic Google Gemini watermark at arbitrary scales (24px to 160px).
    """
    def __init__(self, img_dir=None, total_samples=20000, patch_size=128, external_img_dir=None):
        if external_img_dir is not None and img_dir is None:
            img_dir = external_img_dir
        self.patch_size = patch_size
        self.total_samples = total_samples
        self.image_paths = []

        # Find all real image files in directories
        search_dirs = []
        if img_dir:
            search_dirs.append(img_dir)

        # Standard Kaggle dataset mount locations
        kaggle_standard_dirs = [
            "/kaggle/input/coco-2017-dataset/coco2017/train2017",
            "/kaggle/input/coco-2017-dataset/coco2017/val2017",
            "/kaggle/input/coco-2017-dataset/coco2017/test2017",
            "/kaggle/input/coco2017/val2017",
            "/kaggle/input/coco2017/train2017",
            "/kaggle/input/places365-standard",
            "/kaggle/input/unsplash-dataset",
            "/kaggle/working/real_photos_20k"
        ]
        search_dirs.extend(kaggle_standard_dirs)

        for d in search_dirs:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    for f in files:
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                            self.image_paths.append(os.path.join(root, f))
                            if len(self.image_paths) >= 50000:
                                break
                    if len(self.image_paths) >= 50000:
                        break
            if len(self.image_paths) >= 20000:
                break

        print(f"📸 [RealPhotoDataset] Loaded {len(self.image_paths):,} authentic real-world camera photographs.")

        # Cache canonical watermark templates
        self.cached_sparkles = {
            s: generate_calibrated_gemini_sparkle(s)
            for s in [24, 32, 36, 42, 48, 56, 64, 72, 80, 96, 112, 128]
        }

    def __len__(self):
        return self.total_samples

    def get_sparkle_alpha(self, size: int) -> np.ndarray:
        if size in self.cached_sparkles:
            return self.cached_sparkles[size]
        sparkle = generate_calibrated_gemini_sparkle(size)
        self.cached_sparkles[size] = sparkle
        return sparkle

    def __getitem__(self, idx):
        S = self.patch_size

        # 1. Fetch real camera photograph
        if not self.image_paths:
            raise RuntimeError(
                "❌ No real images found! Please attach COCO 2017, Unsplash, or download real images to run training. "
                "Synthetic images are strictly forbidden."
            )

        img_path = self.image_paths[idx % len(self.image_paths)]
        try:
            with Image.open(img_path) as im:
                im = im.convert('RGB')
                w, h = im.size
                if w >= S and h >= S:
                    rx = random.randint(0, w - S)
                    ry = random.randint(0, h - S)
                    crop = im.crop((rx, ry, rx + S, ry + S))
                else:
                    crop = im.resize((S, S), Image.Resampling.BILINEAR)
                clean = np.array(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0
        except Exception:
            # Fallback to another real image if one file is corrupt
            alt_path = random.choice(self.image_paths)
            with Image.open(alt_path) as im:
                crop = im.convert('RGB').resize((S, S), Image.Resampling.BILINEAR)
                clean = np.array(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0

        # Check luminance for dark background penalty
        lum = 0.299 * clean[0] + 0.587 * clean[1] + 0.114 * clean[2]
        is_dark = float(np.mean(lum)) < 0.30

        # 2. Hard Clean Negative Samples (15% probability: no watermark applied)
        if random.random() < 0.15:
            empty_mask = np.zeros((1, S, S), dtype=np.float32)
            empty_bbox = np.zeros(4, dtype=np.float32)
            conf = np.array([0.0], dtype=np.float32)
            if HAS_TORCH:
                return (
                    torch.from_numpy(clean),
                    torch.from_numpy(clean),
                    torch.from_numpy(empty_mask),
                    torch.from_numpy(empty_bbox),
                    torch.from_numpy(conf),
                    torch.tensor(is_dark, dtype=torch.bool)
                )
            return clean, clean, empty_mask, empty_bbox, conf, is_dark

        # 3. Watermark Scale & Position on Real Photo
        # Scale Invariance: 24px to min(S - 8, 96px)
        max_s = min(S - 8, 96)
        s_size = random.randint(24, max_s)
        base_sparkle = self.get_sparkle_alpha(s_size)

        # Dynamic gain (0.45 to 1.15)
        gain = random.uniform(0.45, 1.15)
        sp_alpha = np.clip(base_sparkle * gain, 0.0, 0.96)

        # Positions: 70% Bottom-Right, 15% Bottom-Left, 15% Random Margins
        corner_roll = random.random()
        margin = random.choice([8, 12, 16, 24, 32, 48])
        if corner_roll < 0.70:
            x0 = S - margin - s_size
            y0 = S - margin - s_size
        elif corner_roll < 0.85:
            x0 = margin
            y0 = S - margin - s_size
        else:
            x0 = random.randint(4, max(4, S - s_size - 4))
            y0 = random.randint(4, max(4, S - s_size - 4))

        x0 = max(0, min(S - s_size, x0))
        y0 = max(0, min(S - s_size, y0))

        # 4. Authentic Linear Optical Alpha Blending: I_wm = I_clean * (1 - alpha) + 1.0 * alpha
        watermarked = clean.copy()
        mask = np.zeros((1, S, S), dtype=np.float32)

        alpha_3d = sp_alpha[np.newaxis, :, :]
        patch_clean = clean[:, y0:y0+s_size, x0:x0+s_size]

        watermarked[:, y0:y0+s_size, x0:x0+s_size] = patch_clean * (1.0 - alpha_3d) + 1.0 * alpha_3d
        mask[0, y0:y0+s_size, x0:x0+s_size] = np.where(sp_alpha > 0.02, 1.0, 0.0)

        # 5. Real Camera ISO Grain & Sensor Noise
        if random.random() > 0.35:
            iso_noise = np.random.normal(0, random.uniform(0.005, 0.018), size=watermarked.shape)
            watermarked = np.clip(watermarked + iso_noise, 0.0, 1.0).astype(np.float32)

        bbox = np.array([
            (x0 + s_size / 2.0) / S,
            (y0 + s_size / 2.0) / S,
            s_size / float(S),
            s_size / float(S)
        ], dtype=np.float32)
        conf = np.array([1.0], dtype=np.float32)

        if HAS_TORCH:
            return (
                torch.from_numpy(watermarked),
                torch.from_numpy(clean),
                torch.from_numpy(mask),
                torch.from_numpy(bbox),
                torch.from_numpy(conf),
                torch.tensor(is_dark, dtype=torch.bool)
            )
        return watermarked, clean, mask, bbox, conf, is_dark


# Backward compatibility alias
Sparkle20kDataset = RealPhotoWatermarkDataset
