"""Validate prepared sequence splits, images, labels, and split disjointness."""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent


def frame_number(path):
    matches = re.findall(r"(\d+)", path.stem)
    return int(matches[-1]) if matches else -1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "dataset")
    root = parser.parse_args().root
    if not root.is_absolute(): root = PROJECT_ROOT / root
    total = 0; errors = []; identifiers = {s: set() for s in ("train", "val", "test")}
    for split in identifiers:
        split_root = root / split
        if not split_root.is_dir(): errors.append(f"{split}: missing directory {split_root}"); print(f"FAIL {split}: directory missing"); continue
        sequences = sorted(p for p in split_root.iterdir() if p.is_dir())
        print(f"{split}: {len(sequences)} sequences")
        for sequence in sequences:
            # Preparation prefixes local order (001_, 002_, ...); remove it to
            # recover the source animation identity across splits.
            identifiers[split].add(re.sub(r"^\d+_", "", sequence.name))
            frame_dir, label_dir = sequence / "frames", sequence / "labels"
            if not frame_dir.is_dir() or not label_dir.is_dir(): errors.append(f"{sequence}: missing frames or labels"); continue
            frames = sorted(frame_dir.glob("*.png"), key=frame_number)
            if not frames: errors.append(f"{sequence}: no PNG frames"); continue
            for frame in frames:
                number = frame_number(frame); labels = [p for p in label_dir.glob("*.json") if frame_number(p) == number]
                if len(labels) != 1: errors.append(f"{sequence}: expected one geometry label for {frame.name}, found {len(labels)}"); continue
                if cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE) is None: errors.append(f"{frame}: unreadable image"); continue
                try:
                    with labels[0].open(encoding="utf-8") as f: label = json.load(f)
                    if not isinstance(label, dict): raise ValueError("label root is not an object")
                    if "source_id" in label: identifiers[split].add(label["source_id"])
                    if not label.get("vertices"): raise ValueError("label has no vertices")
                except Exception as exc: errors.append(f"{labels[0]}: invalid geometry JSON ({exc})"); continue
                total += 1
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = identifiers[left] & identifiers[right]
        if overlap: errors.append(f"sequence leakage: {left}/{right} share {len(overlap)} source IDs")
    print(f"Total validated frames: {total}")
    if errors:
        print("DATASET VALIDATION FAILED")
        for error in errors[:50]: print(f"  - {error}")
        sys.exit(1)
    print("PASS: dataset structure, frame/label pairs and available source IDs")
    print("PASS: sequence-level split construction (preparation samples source sequences without replacement)")
    print("DATASET VALIDATION PASSED")


if __name__ == "__main__": main()
