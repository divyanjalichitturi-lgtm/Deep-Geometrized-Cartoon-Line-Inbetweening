"""Train DeepLine AI from sequence-level ML240 splits."""
import argparse
import csv
from datetime import datetime
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import DEVICE, IMAGE_SIZE, LOSS_WEIGHTS, PROJECT_ROOT
from src.dataset import ML240InbetweenDataset
from src.losses import CombinedLoss
from src.model import create_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "dataset")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-train-samples", type=int, default=512)
    p.add_argument("--max-val-samples", type=int, default=128)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path)
    p.add_argument("--ssim-weight", type=float, default=LOSS_WEIGHTS["ssim"])
    p.add_argument("--mixed-precision", action="store_true", help="Use CUDA automatic mixed precision when CUDA is available")
    return p.parse_args()


def save_curves(history, path):
    """Draw a dependency-light loss plot so training always leaves a PNG."""
    width, height, margin = 900, 480, 55
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    for i in range(6):
        y = margin + i * (height - 2 * margin) // 5
        cv2.line(canvas, (margin, y), (width - margin, y), (224, 229, 235), 1)
    cv2.putText(canvas, "Training and validation loss", (margin, 32), cv2.FONT_HERSHEY_SIMPLEX, .8, (28, 40, 55), 2)
    if history:
        values = [r[k] for r in history for k in ("train_loss", "val_loss") if np.isfinite(r[k])]
        lo, hi = min(values), max(values)
        if hi <= lo: hi = lo + 1e-6
        for key, color in (("train_loss", (208, 108, 41)), ("val_loss", (60, 147, 55))):
            pts = []
            for i, row in enumerate(history):
                x = margin + (width - 2 * margin) * i // max(1, len(history) - 1)
                y = height - margin - int((row[key] - lo) / (hi - lo) * (height - 2 * margin))
                pts.append((x, y))
            for a, b in zip(pts, pts[1:]): cv2.line(canvas, a, b, color, 3, cv2.LINE_AA)
            for q in pts: cv2.circle(canvas, q, 4, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, f"range {lo:.4f} - {hi:.4f}", (margin, height - 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (80, 90, 105), 1)
    cv2.imwrite(str(path), canvas)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1: raise SystemExit("epochs and batch-size must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    root = args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset
    train_all = ML240InbetweenDataset(root / "train", image_size=(IMAGE_SIZE, IMAGE_SIZE))
    val_all = ML240InbetweenDataset(root / "val", image_size=(IMAGE_SIZE, IMAGE_SIZE))
    train_ds = Subset(train_all, range(min(args.max_train_samples, len(train_all))))
    val_ds = Subset(val_all, range(min(args.max_val_samples, len(val_all))))
    pin = DEVICE.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    model = create_model(DEVICE)
    criterion = CombinedLoss(**{**LOSS_WEIGHTS, "ssim": args.ssim_weight})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=2)
    start_epoch, best, stale, history = 1, float("inf"), 0, []
    if args.resume:
        resume_path = args.resume if args.resume.is_absolute() else PROJECT_ROOT / args.resume
        state = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        if not isinstance(state, dict) or "model_state_dict" not in state:
            raise RuntimeError("Resume file is not a full DeepLine training checkpoint; old weights-only checkpoints cannot resume")
        try:
            model.load_state_dict(state["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"Resume checkpoint architecture is incompatible: {exc}") from exc
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scheduler_state_dict"): scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best = float(state.get("validation_loss", float("inf")))
        history = state.get("history", [])
        print(f"Resuming at epoch {start_epoch}; best validation loss {best:.6f}")
    model_dir = PROJECT_ROOT / "models"; output = PROJECT_ROOT / "outputs"
    checkpoint_root = output / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True); checkpoint_root.mkdir(parents=True, exist_ok=True)
    for name in ("generated", "comparisons", "metrics"): (output / name).mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best_model.pth"
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if args.resume:
        candidate_dir = resume_path.parent
        try: candidate_dir.relative_to(checkpoint_root)
        except ValueError: pass
        else: run_name = candidate_dir.name
    else:
        base_name = run_name; suffix = 1
        while (checkpoint_root / run_name).exists():
            run_name = f"{base_name}_{suffix}"; suffix += 1
    checkpoint_dir = checkpoint_root / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if best_path.exists() and not args.resume:
        try:
            old = torch.load(best_path, map_location="cpu", weights_only=False)
            old_state = old.get("model_state_dict", old) if isinstance(old, dict) else old
            model.load_state_dict(old_state, strict=True)
            print("Existing checkpoint matches this model; training a fresh run will replace it only on validation improvement.")
        except Exception:
            archived = checkpoint_root / f"legacy_best_model_incompatible_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            if not archived.exists(): shutil.copy2(best_path, archived)
            print(f"Archived incompatible previous checkpoint at {archived}; starting fresh.")
    use_amp = args.mixed_precision and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Device: {DEVICE}; mixed precision: {use_amp}; train samples: {len(train_ds)}; val samples: {len(val_ds)}")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); train_sum = 0.
        for batch in train_dl:
            x, y = batch["input"].to(DEVICE, non_blocking=pin), batch["target"].to(DEVICE, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp): pred = model(x); loss, _ = criterion(pred, y)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); train_sum += float(loss.detach())
        model.eval(); val_sum = 0.
        with torch.no_grad():
            for batch in val_dl:
                x, y = batch["input"].to(DEVICE, non_blocking=pin), batch["target"].to(DEVICE, non_blocking=pin)
                loss, _ = criterion(model(x), y); val_sum += float(loss)
        train_loss, val_loss = train_sum / max(1, len(train_dl)), val_sum / max(1, len(val_dl))
        scheduler.step(val_loss)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row); print(f"Epoch {epoch}/{args.epochs} | train {train_loss:.6f} | val {val_loss:.6f} | lr {row['learning_rate']:.2g}")
        payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "validation_loss": val_loss, "configuration": vars(args), "architecture": model.architecture, "history": history}
        torch.save(payload, checkpoint_dir / f"epoch_{epoch:03d}.pth")
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save(payload, best_path)
        else: stale += 1
        with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        save_curves(history, output / "training_curves.png")
        if stale >= args.patience:
            print(f"Early stopping: validation did not improve for {args.patience} epochs."); break
    print(f"Best validation loss: {best:.6f}; checkpoint: {best_path}")


if __name__ == "__main__": main()
