#!/usr/bin/env python3
"""
Model Training Script for ClearSpark AI
Generates synthetic dataset and trains the WAL-Net and CIS-Net architectures.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training.trainer import WatermarkAITrainer

def main(epochs=None, steps=None):
    if epochs is None or steps is None:
        import argparse
        parser = argparse.ArgumentParser(description="Train ClearSpark AI")
        parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
        parser.add_argument("--steps", type=int, default=15, help="Steps per epoch")
        args, _ = parser.parse_known_args()
        num_epochs = args.epochs
        steps_per_epoch = args.steps
    else:
        num_epochs = epochs
        steps_per_epoch = steps

    print("=" * 65)
    print("   CLEARSPARK AI — DEEP LEARNING MODEL TRAINING PIPELINE")
    print("=" * 65)

    print("\n[1/3] Initializing Neural Network Architectures...")
    trainer = WatermarkAITrainer(lr=0.001)

    print(f"[2/3] Training WAL-Net & CIS-Net ({num_epochs} Epochs, {steps_per_epoch} Steps/Epoch)...")
    start_t = time.time()

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        wal_loss, cis_loss = trainer.train_epoch(num_steps=steps_per_epoch, batch_size=2)
        print(f"Epoch {epoch + 1} Summary: WAL Loss = {wal_loss:.4f}, CIS Loss = {cis_loss:.4f}")

    total_time = time.time() - start_t
    print(f"\n[3/3] Training finished in {total_time:.2f}s! Saving learned weights...")
    trainer.save_checkpoints()

    # Re-export web weights automatically
    try:
        from export_web_weights import export_weights
        base_dir = os.path.dirname(os.path.abspath(__file__))
        w_dir = os.path.join(base_dir, "weights")
        out_file = os.path.join(w_dir, "sparklenet_web_weights.json")
        export_weights(w_dir, out_file)
    except Exception as e:
        print(f"Web export note: {e}")

    print("\n" + "=" * 65)
    print("   ALL WEIGHTS & METADATA SAVED SUCCESSFULLY TO /weights/")
    print("=" * 65)

if __name__ == "__main__":
    main()
