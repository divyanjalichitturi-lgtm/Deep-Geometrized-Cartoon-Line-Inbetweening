"""Inspect uploaded-image inference on a known held-out endpoint pair."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import DEVICE, MODEL_PATH, PROJECT_ROOT
from src.dataset import ML240InbetweenDataset
from src.inference import load_trained_model, predict_image_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    test_set = ML240InbetweenDataset(PROJECT_ROOT / "dataset" / "test")
    if not 0 <= args.index < len(test_set):
        raise SystemExit(f"index must be between 0 and {len(test_set) - 1}")
    sample = test_set.samples[args.index]
    frame_a_path, _, target_path, _, frame_b_path, _ = sample
    model, checkpoint = load_trained_model(args.model, DEVICE)
    with Image.open(frame_a_path) as image:
        frame_a = image.convert("RGB")
    with Image.open(frame_b_path) as image:
        frame_b = image.convert("RGB")
    # Same estimated-geometry preprocessing and inference call used by Streamlit.
    result = predict_image_pair(model, frame_a, frame_b, DEVICE)
    raw = result["prediction"].squeeze().numpy()
    generated = result["generated_frame"]
    train_input = test_set[args.index]["input"].numpy()
    upload_input = result["model_input"][0].numpy()
    channel_names = ("line A", "vertex A", "graph A", "line B", "vertex B", "graph B", "t")
    print("Channel distribution comparison (training ML240 input vs uploaded-image inference):")
    for index, channel_name in enumerate(channel_names):
        for source_name, values in (("TRAIN", train_input[index]), ("UPLOAD", upload_input[index])):
            print(
                f"  {channel_name:8s} {source_name:6s} shape={values.shape} dtype={values.dtype} "
                f"min={values.min():.5f} max={values.max():.5f} mean={values.mean():.5f} "
                f"std={values.std():.5f} nonzero%={(values != 0).mean() * 100:.3f}"
            )
    out = PROJECT_ROOT / "outputs" / "generated"
    comparisons = PROJECT_ROOT / "outputs" / "comparisons"
    out.mkdir(parents=True, exist_ok=True)
    comparisons.mkdir(parents=True, exist_ok=True)
    stem = f"diagnostic_{args.index:04d}"
    raw_path = out / f"{stem}_raw_prediction.npy"
    display_path = out / f"{stem}_display.png"
    comparison_path = comparisons / f"{stem}_comparison.png"
    np.save(raw_path, raw)
    cv2.imwrite(str(display_path), generated)
    target = Image.open(target_path).convert("L")
    middle = np.asarray(target.resize(generated.shape[::-1], Image.Resampling.LANCZOS))
    tile_a = np.asarray(frame_a.convert("L").resize(generated.shape[::-1], Image.Resampling.LANCZOS))
    tile_b = np.asarray(frame_b.convert("L").resize(generated.shape[::-1], Image.Resampling.LANCZOS))
    cv2.imwrite(str(comparison_path), np.concatenate([tile_a, generated, middle, tile_b], axis=1))
    print(f"Checkpoint: {args.model.resolve()}")
    print(f"Architecture: {model.architecture}; strict load: PASS; epoch: {checkpoint.get('epoch')}")
    print(f"Inference mode: {result.get('geometry_mode', 'unknown')}")
    print(f"Raw: shape={raw.shape} dtype={raw.dtype} min={raw.min():.6f} max={raw.max():.6f} mean={raw.mean():.6f} std={raw.std():.6f}")
    print("Raw prediction value bins (%): " + str(result["prediction_stats"]["value_bins_pct"]))
    print(f"PNG: mode=L size={generated.shape[1]}x{generated.shape[0]} pixel_min={generated.min()} pixel_max={generated.max()}")
    print(f"Raw prediction: {raw_path}")
    print(f"Display prediction: {display_path}")
    print(f"Comparison: {comparison_path}")


if __name__ == "__main__":
    main()
