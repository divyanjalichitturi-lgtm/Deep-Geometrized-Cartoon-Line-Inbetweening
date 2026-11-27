"""Evaluate a selected held-out ML240 triple and save report-ready artifacts."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from config import DEVICE, IMAGE_SIZE, MODEL_PATH, PROJECT_ROOT
from src.dataset import ML240InbetweenDataset
from src.inference import load_trained_model, predict_model_input
from src.metrics import metrics
from src.preprocess import tensor_to_display_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    dataset_root = args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset
    model_path = args.model if args.model.is_absolute() else PROJECT_ROOT / args.model
    ds = ML240InbetweenDataset(dataset_root / "test", image_size=(IMAGE_SIZE, IMAGE_SIZE))
    if not 0 <= args.index < len(ds): raise SystemExit(f"index must be between 0 and {len(ds)-1}")
    if not model_path.is_file(): raise SystemExit(f"Model checkpoint not found: {model_path}. Train the model first.")
    try:
        model, checkpoint = load_trained_model(model_path, DEVICE)
    except Exception as exc:
        raise SystemExit(f"Cannot load checkpoint with current architecture: {exc}") from exc
    item = ds[args.index]; x = item["input"].unsqueeze(0).to(DEVICE); target = item["target"].squeeze().numpy()
    inference = predict_model_input(model, x, DEVICE)
    prediction = inference["prediction"]
    elapsed = inference["inference_seconds"]
    pred = prediction.squeeze().numpy()
    result = metrics(pred, target)
    out = PROJECT_ROOT / "outputs"
    gen_dir, comp_dir, metric_dir = out / "generated", out / "comparisons", out / "metrics"
    for directory in (gen_dir, comp_dir, metric_dir): directory.mkdir(parents=True, exist_ok=True)
    generated = inference["generated_frame"]
    print(
        f"Prediction: shape={tuple(prediction.shape)} dtype={prediction.dtype} "
        f"min={prediction.min().item():.6f} max={prediction.max().item():.6f} "
        f"mean={prediction.mean().item():.6f} std={prediction.std(unbiased=False).item():.6f}"
    )
    print("Prediction value bins (%): " + json.dumps(inference["prediction_stats"]["value_bins_pct"], sort_keys=True))
    print(
        f"Generated PNG: mode=L size={generated.shape[1]}x{generated.shape[0]} "
        f"pixel_min={generated.min()} pixel_max={generated.max()} "
        f"pixel_mean={generated.mean():.3f}"
    )
    truth = tensor_to_display_image(torch.from_numpy(target)[None, None])
    start_img = tensor_to_display_image(x[0, 0:1].cpu()); end_img = tensor_to_display_image(x[0, 3:4].cpu())
    stem = f"test_{args.index:04d}"
    cv2.imwrite(str(gen_dir / f"{stem}_generated.png"), generated)
    cv2.imwrite(str(gen_dir / f"{stem}_ground_truth.png"), truth)
    tiles = []
    for label, img in (("START FRAME", start_img), ("GENERATED MIDDLE", generated), ("GROUND TRUTH", truth), ("END FRAME", end_img)):
        tile = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .42, (20, 75, 160), 1, cv2.LINE_AA)
        tiles.append(tile)
    comparison = np.concatenate(tiles, axis=1)
    cv2.imwrite(str(comp_dir / f"{stem}_comparison.png"), comparison)
    report = {**result, "inference_seconds": elapsed, "test_index": args.index, "model": str(model_path), "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None}
    with (metric_dir / f"{stem}_metrics.json").open("w", encoding="utf-8") as f: json.dump(report, f, indent=2, allow_nan=False)
    with (metric_dir / "latest_metrics.json").open("w", encoding="utf-8") as f: json.dump(report, f, indent=2, allow_nan=False)
    csv_path = metric_dir / "latest_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report)); writer.writeheader(); writer.writerow(report)
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Comparison: {comp_dir / f'{stem}_comparison.png'}")


if __name__ == "__main__": main()
