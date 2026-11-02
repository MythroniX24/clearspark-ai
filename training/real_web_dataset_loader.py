"""
Real Web Dataset Loader
Loads paired (watermarked, clean, mask) samples from real 1024x1024 web photos.
Extracts multi-scale contextual ROIs for high-performance memory-safe training.
"""

import os
import glob
import numpy as np
from PIL import Image

class RealWebDatasetLoader:
    def __init__(self, data_dir=None, patch_size=96):
        base_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "real_web_dataset"
        )
        self.clean_dir = os.path.join(base_dir, "clean")
        self.wm_dir = os.path.join(base_dir, "watermarked")
        self.mask_dir = os.path.join(base_dir, "masks")
        self.patch_size = patch_size

        self.samples = sorted(glob.glob(os.path.join(self.clean_dir, "real_*.jpg")))
        self.indices = list(range(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def get_sample(self, idx=None):
        if idx is None:
            idx = int(np.random.randint(0, len(self.samples)))

        clean_path = self.samples[idx]
        fname = os.path.basename(clean_path)
        wm_path = os.path.join(self.wm_dir, fname)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(fname)[0] + ".png")

        im_clean = Image.open(clean_path).convert("RGB")
        im_wm = Image.open(wm_path).convert("RGB")
        im_mask = Image.open(mask_path).convert("L")

        W, H = im_clean.size
        S = self.patch_size

        # Find watermark location from mask
        mask_arr = np.array(im_mask)
        ys, xs = np.where(mask_arr > 50)

        if len(ys) > 0:
            # Center crop around watermark with context
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            # Add random jitter to context center
            jitter = 12
            cx += int(np.random.randint(-jitter, jitter + 1))
            cy += int(np.random.randint(-jitter, jitter + 1))
            x0 = max(0, min(W - S, cx - S // 2))
            y0 = max(0, min(H - S, cy - S // 2))
        else:
            # Hard negative: pick random corner or texture crop
            x0 = int(np.random.choice([0, max(0, W - S), np.random.randint(0, max(1, W - S))]))
            y0 = int(np.random.choice([0, max(0, H - S), np.random.randint(0, max(1, H - S))]))

        crop_clean = im_clean.crop((x0, y0, x0 + S, y0 + S))
        crop_wm = im_wm.crop((x0, y0, x0 + S, y0 + S))
        crop_mask = im_mask.crop((x0, y0, x0 + S, y0 + S))

        t_clean = (np.array(crop_clean).transpose(2, 0, 1) / 255.0).astype(np.float32)
        t_wm = (np.array(crop_wm).transpose(2, 0, 1) / 255.0).astype(np.float32)
        t_mask = (np.array(crop_mask)[np.newaxis, :, :] / 255.0).astype(np.float32)

        # Bounding box
        if len(ys) > 0:
            min_x = max(0, int(np.min(xs)) - x0)
            max_x = min(S, int(np.max(xs)) - x0)
            min_y = max(0, int(np.min(ys)) - y0)
            max_y = min(S, int(np.max(ys)) - y0)
            bw = (max_x - min_x) / float(S)
            bh = (max_y - min_y) / float(S)
            bx = (min_x + max_x) / (2.0 * S)
            by = (min_y + max_y) / (2.0 * S)
            bbox = np.array([bx, by, bw, bh], dtype=np.float32)
        else:
            bbox = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        return t_wm, t_clean, t_mask, bbox

    def get_batch(self, batch_size=2):
        w_list, c_list, m_list, b_list = [], [], [], []
        for _ in range(batch_size):
            w, c, m, b = self.get_sample()
            w_list.append(w)
            c_list.append(c)
            m_list.append(m)
            b_list.append(b)
        return (
            np.stack(w_list, axis=0),
            np.stack(c_list, axis=0),
            np.stack(m_list, axis=0),
            np.stack(b_list, axis=0)
        )
