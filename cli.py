#!/usr/bin/env python3
"""
CLI Tool for Listora Watermark AI
Usage:
    python3 cli.py clean <input_image> [output_image]
    python3 cli.py detect <input_image>
    python3 cli.py train
    python3 cli.py benchmark [image_dir]
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cleaner_ai import ClearSparkAI

def main():
    parser = argparse.ArgumentParser(description="ClearSpark AI — Deep Watermark Removal CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean watermark from an image")
    clean_parser.add_argument("input", help="Path to input image")
    clean_parser.add_argument("-o", "--output", help="Path to output image (optional)")
    clean_parser.add_argument("--neural", action="store_true", help="Use pure Gated Neural Network inpainting (CIS-Net)")

    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect watermark position and size")
    detect_parser.add_argument("input", help="Path to input image")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument("--epochs", type=int, default=3, help="Epochs")
    train_parser.add_argument("--steps", type=int, default=15, help="Steps per epoch")

    # Download Dataset command
    dl_parser = subparsers.add_parser("download-data", help="Download real full web photographs and build dataset")
    dl_parser.add_argument("--count", type=int, default=100, help="Number of real full photos to download")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark across image folder")
    bench_parser.add_argument("dir", nargs="?", default="/storage/emulated/0/Download/Gemini_Samples/benchmark_100", help="Directory of test images")

    args = parser.parse_args()

    if args.command == "clean":
        ai = ClearSparkAI()
        out_path = args.output or (os.path.splitext(args.input)[0] + "_cleaned.png")
        mode_str = "Gated Neural Network (CIS-Net)" if args.neural else "Fast Optical De-Blending"
        print(f"[*] Processing {args.input} with ClearSpark AI [{mode_str}]...")
        res = ai.clean_image(args.input, out_path, use_neural_inpaint=args.neural)
        if res['success']:
            print(f"[+] SUCCESS! {res['note']} ({res['elapsed_ms']} ms)")
            print(f"[+] Output saved to: {out_path}")
        else:
            print(f"[-] Result: {res['note']} ({res['elapsed_ms']} ms)")

    elif args.command == "detect":
        from PIL import Image
        import numpy as np
        ai = ClearSparkAI()
        im = Image.open(args.input).convert('RGB')
        hit = ai.detect_watermark(np.array(im))
        if hit and hit['confidence'] >= 0.22:
            print(f"[+] Detected Watermark: {hit['side']}px at ({hit['x']}, {hit['y']}) in {hit['corner'].upper()} corner")
            print(f"    Confidence: {hit['confidence']*100:.2f}%")
        else:
            print("[-] No watermark detected.")

    elif args.command == "train":
        from train_model import main as run_train
        run_train(epochs=args.epochs, steps=args.steps)

    elif args.command == "download-data":
        from training.build_real_dataset import build_dataset
        build_dataset(num_images=args.count)

    elif args.command == "benchmark":
        import glob
        ai = ClearSparkAI()
        images = glob.glob(os.path.join(args.dir, "*.png")) + glob.glob(os.path.join(args.dir, "*.jpg"))
        print(f"[*] Benchmarking {len(images)} images in {args.dir}...")
        passed = 0
        total_time = 0.0

        for idx, img_path in enumerate(images[:20]):
            t0 = time.time()
            res = ai.clean_image(img_path)
            dt = time.time() - t0
            total_time += dt
            if res['success']:
                passed += 1
                status = f"CLEANED ({res['hit']['side']}px, {res['hit']['confidence']*100:.1f}%)"
            else:
                status = "CLEAN / NO WM"
            print(f"[{idx+1:02d}/{len(images[:20])}] {os.path.basename(img_path)}: {status} ({int(dt*1000)}ms)")

        print(f"\n[+] Benchmark Complete: {passed}/{len(images[:20])} processed in {total_time:.2f}s ({total_time/max(1, len(images[:20]))*1000:.1f}ms/img)")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
