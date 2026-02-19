"""
train.py – train the raga classifier on the annotated dataset.

Usage
─────
  python train.py              # train on all usable files
  python train.py --verbose    # show per-file extraction progress (default)
  python train.py --quiet      # suppress per-file messages
"""

import argparse
import sys
import time

# make sure the project root is on sys.path when called from anywhere
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.data_loader import load_dataset
from pipeline.model import train


def main():
    parser = argparse.ArgumentParser(
        description="Train the Svarai raga classifier."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-file extraction messages."
    )
    args = parser.parse_args()
    verbose = not args.quiet

    print("=" * 60)
    print("  Svarai – Raga Accuracy Pipeline  |  Training")
    print("=" * 60)

    # ── Step 1: load & extract features ─────────────────────────────────────
    print("\n[1/2] Extracting features from dataset …\n")
    t0 = time.time()
    X, y, weights, labels = load_dataset(verbose=verbose)
    t1 = time.time()
    print(f"\nFeature extraction done in {t1 - t0:.1f}s")

    # ── Step 2: train model ───────────────────────────────────────────────────
    print("\n[2/2] Training model …")
    results = train(X, y, weights, labels, verbose=True)
    t2 = time.time()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Ragas : {results['ragas']}")
    print(f"  Samples : {results['n_samples']}")
    print(f"  CV Accuracy : {results['cv_accuracy_mean']:.3f} "
          f"± {results['cv_accuracy_std']:.3f}")
    print(f"  CV F1-macro : {results['cv_f1_mean']:.3f} "
          f"± {results['cv_f1_std']:.3f}")
    print(f"  Total time  : {t2 - t0:.1f}s")
    print("=" * 60)
    print("\nNow run  python evaluate.py  to score a recording.\n")


if __name__ == "__main__":
    main()
