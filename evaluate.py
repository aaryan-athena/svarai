"""
evaluate.py – score a user's singing against a target raga.

Usage
─────
  python evaluate.py <audio_file> <raga>

  audio_file : path to a WAV (or any librosa-supported format) recording
  raga       : name of the raga the user intended to sing
               (case-insensitive; Asavari | Sarang | Yaman)

Examples
────────
  python evaluate.py my_recording.wav Yaman
  python evaluate.py recordings/test.wav asavari
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.evaluator import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate singing accuracy for a given raga.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio_file", help="Path to the user's audio recording.")
    parser.add_argument("raga",       help="Target raga name (e.g. Yaman, Asavari, Sarang).")
    args = parser.parse_args()

    if not os.path.isfile(args.audio_file):
        print(f"ERROR: File not found: {args.audio_file}")
        sys.exit(1)

    print("=" * 60)
    print("  Svarai – Raga Accuracy Pipeline  |  Evaluation")
    print("=" * 60)
    print(f"\n  Audio  : {args.audio_file}")
    print(f"  Target : {args.raga}\n")
    print("Extracting features …", flush=True)

    try:
        result = evaluate(args.audio_file, args.raga)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    _print_result(result)


def _print_result(r: dict):
    correct_tag = "✓ CORRECT" if r["correct"] else "✗ WRONG"
    print("=" * 60)
    print(f"  Result          : {correct_tag}")
    print(f"  Target raga     : {r['target_raga']}")
    print(f"  Predicted raga  : {r['predicted_raga']}")
    print(f"  Overall score   : {r['score_pct']} / 100")
    print("-" * 60)
    print(f"  SVM confidence  : {r['svm_confidence'] * 100:.1f} %")
    print(f"  DTW similarity  : {r['dtw_similarity'] * 100:.1f} %")
    print("-" * 60)
    print("  Per-raga probabilities:")
    for raga, prob in sorted(r["class_probs"].items(),
                             key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"    {raga:<12} {prob*100:5.1f}%  {bar}")
    print("=" * 60)
    print("\nFeedback:\n")
    for line in r["feedback"].split("\n"):
        print(f"  {line}")
    print()


if __name__ == "__main__":
    main()
