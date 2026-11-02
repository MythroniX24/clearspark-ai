"""
Authentic Gemini Real Web Dataset Builder & Compositor.
Composites authentic Google Gemini / Imagen 3 watermark profiles (from calibrated bg_48.png/bg_96.png)
onto genuine 1024x1024 web photographs using official catalog margins.
"""

import os
import sys
import glob
import time
from typing import Optional
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.gemini_alpha_provider import get_official_gemini_sparkle

def build_dataset(num_images: int = 200, out_dir: Optional[str] = None) -> None:
    base_dir = out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "real_web_dataset"
    )
    clean_dir = os.path.join(base_dir, "clean")
    wm_dir = os.path.join(base_dir, "watermarked")
    mask_dir = os.path.join(base_dir, "masks")

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(wm_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print("=" * 70)
    print("   BUILDING 200 AUTHENTIC GEMINI WATERMARKED REAL PHOTOGRAPHS")
    print(f"   Target: {num_images} Images (1024x1024) -> {base_dir}")
    print("   Watermark: Official Google Gemini Sparkle (bg_48 calibrated)")
    print("   Margins: Catalog Anchors (96px large margin, 32px small margin)")
    print("=" * 70)

    clean_files = sorted(glob.glob(os.path.join(clean_dir, "real_*.jpg")))
    if not clean_files:
        print(f"   [!] No clean source images found in {clean_dir}")
        return

    start_t = time.time()
    processed = 0

    # Official discrete Gemini anchor presets for 1024x1024:
    # 70% Imagen 3 current standard (48px, m96)
    # 25% Classic Gemini (48px, m32)
    # 5% v2 Small (36px, m96)
    anchor_presets = [
        {'size': 48, 'margin': 96, 'weight': 0.70},
        {'size': 48, 'margin': 32, 'weight': 0.25},
        {'size': 36, 'margin': 96, 'weight': 0.05},
    ]

    for idx, clean_path in enumerate(clean_files[:num_images]):
        try:
            im_clean = Image.open(clean_path).convert("RGB")
            W, H = im_clean.size
            clean_arr = np.array(im_clean).astype(np.float32)

            # 10% Hard negatives (clean images without any watermark)
            is_negative = (idx % 10 == 9)

            wm_img = clean_arr.copy()
            mask = np.zeros((H, W), dtype=np.uint8)

            if not is_negative:
                # Select official anchor
                r = np.random.rand()
                if r < 0.70:
                    cfg = anchor_presets[0]
                elif r < 0.95:
                    cfg = anchor_presets[1]
                else:
                    cfg = anchor_presets[2]

                s = cfg['size']
                margin = cfg['margin']

                if W < s or H < s:
                    continue

                # Bottom-Right placement
                x0 = W - margin - s
                y0 = H - margin - s
                x0 = max(0, min(W - s, x0))
                y0 = max(0, min(H - s, y0))

                # Load official calibrated Gemini sparkle
                gain = float(np.random.uniform(0.75, 1.05))
                sp_alpha = get_official_gemini_sparkle(s, alpha_gain=gain)
                sp_expanded = sp_alpha[:, :, np.newaxis]

                # Optical linear alpha blend: I_wm = I_clean * (1 - alpha) + 255 * alpha
                patch = wm_img[y0:y0+s, x0:x0+s, :]
                blended = patch * (1.0 - sp_expanded) + 255.0 * sp_expanded
                wm_img[y0:y0+s, x0:x0+s, :] = blended

                # Ground-truth binary mask (thresholded at low alpha noise floor)
                mask[y0:y0+s, x0:x0+s] = np.where(sp_alpha > 0.02, 255, 0).astype(np.uint8)

            # Save watermarked photo
            base_name = os.path.basename(clean_path)
            wm_path = os.path.join(wm_dir, base_name)
            wm_pil = Image.fromarray(np.clip(wm_img, 0, 255).astype(np.uint8))
            
            # Realistic JPEG compression
            q = int(np.random.randint(82, 95)) if np.random.rand() > 0.3 else 95
            wm_pil.save(wm_path, "JPEG", quality=q)

            # Save mask
            mask_name = os.path.splitext(base_name)[0] + ".png"
            mask_path = os.path.join(mask_dir, mask_name)
            Image.fromarray(mask).save(mask_path, "PNG")

            processed += 1
            if processed % 25 == 0 or processed == num_images:
                elapsed = time.time() - start_t
                tag = "[Negative]" if is_negative else f"[Gemini {s}px @ m{margin}]"
                print(f"[{processed:03d}/{num_images}] Processed {base_name} {tag} ({elapsed:.1f}s)")

        except Exception as e:
            print(f"Error processing {clean_path}: {e}")

    total_time = time.time() - start_t
    print("\n" + "=" * 70)
    print(f"   DATASET BUILD COMPLETE: {processed} AUTHENTIC GEMINI WATERMARKED PHOTOS")
    print(f"   Clean Images:       {clean_dir}")
    print(f"   Watermarked Images: {wm_dir}")
    print(f"   Ground Truth Masks: {mask_dir}")
    print(f"   Total Build Time:   {total_time:.2f}s")
    print("=" * 70)

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    build_dataset(count)
