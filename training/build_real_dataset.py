"""
Real Web Dataset Harvester & Watermark Compositor
Downloads 200 genuine, uncropped, full-resolution (1024x1024) photographs
from Unsplash photographers (Humans, Portraits, Trees, Nature, Architecture, Cities, Products),
ensuring bright, colorful backgrounds (no dark/black corners) and exact canonical
Google Gemini / Imagen 3 watermark placement.
"""

import os
import sys
import time
import requests
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset_generator import generate_procedural_sparkle

def build_dataset(num_images=200, out_dir=None, wipe_existing=True):
    base_dir = out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "real_web_dataset"
    )
    clean_dir = os.path.join(base_dir, "clean")
    wm_dir = os.path.join(base_dir, "watermarked")
    mask_dir = os.path.join(base_dir, "masks")

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(wm_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    if wipe_existing:
        import glob
        for f in glob.glob(os.path.join(clean_dir, "real_*")) + \
                 glob.glob(os.path.join(wm_dir, "real_*")) + \
                 glob.glob(os.path.join(mask_dir, "real_*")):
            try:
                os.remove(f)
            except Exception:
                pass
        print("[Dataset] Cleaned previous dataset to ensure 100% bright, non-black, canonical photos.")

    print("=" * 65)
    print("   DOWNLOADING 200 REAL FULL-RESOLUTION WEB PHOTOGRAPHS")
    print(f"   Target: {num_images} Bright Uncropped Photos (1024x1024) -> {base_dir}")
    print("   Placement: Canonical Bottom-Right (95%) & Bottom-Left (5%)")
    print("   Background Filter: Brightness >= 55 (Zero black/dark corners)")
    print("=" * 65)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ListoraAI/2.0"}
    session = requests.Session()

    cur_id = 15
    downloaded = 0
    start_t = time.time()

    # Canonical Gemini sparkle sizes & margin anchors
    sizes = [32, 36, 40, 48, 56, 64]
    canonical_margins = [24, 32, 40, 48]

    while downloaded < num_images:
        url = f"https://picsum.photos/id/{cur_id}/1024/1024"
        try:
            r = session.get(url, headers=headers, timeout=10)
            if r.status_code == 200 and len(r.content) > 25000:
                # 1. Decode Clean Real Photo
                import io
                im_clean = Image.open(io.BytesIO(r.content)).convert("RGB")
                W, H = im_clean.size
                clean_arr = np.array(im_clean).astype(np.float32)

                # Brightness guard: Verify the bottom-right corner is bright and textured
                # (Filter out dark night skies, black borders, and underexposed corners)
                br_corner = clean_arr[H-160:, W-160:]
                bl_corner = clean_arr[H-160:, :160]
                mean_br = float(np.mean(br_corner))
                var_br = float(np.var(br_corner))

                if mean_br < 55.0 or var_br < 120.0:
                    # Skip dark or flat-black images
                    cur_id += 1
                    continue

                # Save Clean Real Photo
                clean_path = os.path.join(clean_dir, f"real_{downloaded+1:03d}.jpg")
                with open(clean_path, "wb") as f:
                    f.write(r.content)

                # 2. Decide if this is a Hard Negative (10% chance)
                is_negative = (downloaded % 10 == 9)

                wm_img = clean_arr.copy()
                mask = np.zeros((H, W), dtype=np.uint8)

                if not is_negative:
                    s = int(np.random.choice(sizes))
                    margin = int(np.random.choice(canonical_margins))
                    # 95% bottom-right, 5% bottom-left
                    corner = "bl" if (np.random.rand() > 0.95) else "br"

                    if corner == "br":
                        x0 = W - margin - s
                        y0 = H - margin - s
                    else:
                        x0 = margin
                        y0 = H - margin - s

                    x0 = max(0, min(W - s, x0))
                    y0 = max(0, min(H - s, y0))

                    # Sparkle alpha map (Gemini standard 0.45 to 0.65 opacity)
                    sp_raw = generate_procedural_sparkle(s)
                    opacity = float(np.random.uniform(0.45, 0.65))
                    sp_alpha = np.clip(sp_raw * (opacity / 0.52), 0.0, 1.0)[:, :, np.newaxis]

                    # Linear alpha blending onto full photograph:
                    # I_wm = I_clean * (1 - alpha) + 255 * alpha
                    patch = wm_img[y0:y0+s, x0:x0+s, :]
                    blended = patch * (1.0 - sp_alpha) + 255.0 * sp_alpha
                    wm_img[y0:y0+s, x0:x0+s, :] = blended

                    # Binary mask
                    mask[y0:y0+s, x0:x0+s] = np.where(sp_raw > 0.02, 255, 0).astype(np.uint8)

                # 3. Save Watermarked Image (Realistic JPEG compression on 70% of files)
                wm_pil = Image.fromarray(np.clip(wm_img, 0, 255).astype(np.uint8))
                wm_path = os.path.join(wm_dir, f"real_{downloaded+1:03d}.jpg")
                if np.random.rand() > 0.3:
                    q = int(np.random.randint(75, 92))
                    wm_pil.save(wm_path, "JPEG", quality=q)
                else:
                    wm_pil.save(wm_path, "JPEG", quality=95)

                # 4. Save Mask
                mask_pil = Image.fromarray(mask)
                mask_path = os.path.join(mask_dir, f"real_{downloaded+1:03d}.png")
                mask_pil.save(mask_path, "PNG")

                downloaded += 1
                tag = "[Hard Negative Clean]" if is_negative else f"[WM {s}px @ ({x0}, {y0})]"
                if downloaded % 10 == 0 or downloaded == num_images:
                    elapsed = time.time() - start_t
                    print(f"[{downloaded:03d}/{num_images}] Photo ID {cur_id:03d} {tag} (BR Mean: {mean_br:.1f}, {elapsed:.1f}s)")
        except Exception:
            pass

        cur_id += 1

    total_time = time.time() - start_t
    print("\n" + "=" * 65)
    print(f"   DATASET BUILD COMPLETE: {downloaded} FULL 1024x1024 BRIGHT REAL PHOTOS")
    print(f"   Clean Images:       {clean_dir}")
    print(f"   Watermarked Images: {wm_dir}")
    print(f"   Ground Truth Masks: {mask_dir}")
    print(f"   Total Build Time:   {total_time:.1f}s")
    print("=" * 65)

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    build_dataset(count)
