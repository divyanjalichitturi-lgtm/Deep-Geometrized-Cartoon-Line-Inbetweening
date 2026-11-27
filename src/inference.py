"""Shared strict checkpoint loading and forward inference for CLI and UI."""
from pathlib import Path
import math
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

from .model import create_model
from .preprocess import image_to_inputs, restore_aspect, tensor_to_display_image


def validate_inference_result(result: object) -> dict[str, Any]:
    """Validate the dictionary contract returned by :func:`predict_image_pair`."""
    if not isinstance(result, dict):
        raise TypeError(
            "Inference returned an invalid result: expected a dictionary, "
            f"received {type(result).__name__}. Restart Streamlit to load the current inference code."
        )
    required = {"prediction", "generated_frame", "inference_seconds", "prediction_stats", "image_stats", "model_input"}
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"Inference result is missing required key(s): {', '.join(missing)}")
    if not isinstance(result["prediction"], torch.Tensor):
        raise TypeError("Inference result key 'prediction' must contain a torch.Tensor")
    prediction = result["prediction"]
    if prediction.ndim != 4 or prediction.shape[1] != 1 or not torch.isfinite(prediction).all():
        raise ValueError("Inference result key 'prediction' must be a finite BCHW tensor with one grayscale channel")
    frame = result["generated_frame"]
    if not isinstance(frame, np.ndarray) or frame.ndim != 2 or frame.dtype != np.uint8:
        raise TypeError("Inference result key 'generated_frame' must be a 2-D uint8 grayscale image")
    seconds = result["inference_seconds"]
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Inference result key 'inference_seconds' must be a finite non-negative number")
    stat_keys = {"shape", "dtype", "min", "max", "mean", "std"}
    stats = result["prediction_stats"]
    if not isinstance(stats, dict) or not stat_keys.issubset(stats):
        raise ValueError("Inference result key 'prediction_stats' is missing required statistics")
    if tuple(stats["shape"]) != tuple(prediction.shape) or stats["dtype"] != str(prediction.dtype):
        raise ValueError("Inference result 'prediction_stats' shape/dtype do not match 'prediction'")
    if any(not isinstance(stats[key], (int, float)) or not math.isfinite(stats[key]) for key in ("min", "max", "mean", "std")):
        raise ValueError("Inference result 'prediction_stats' must contain finite numeric values")
    image_stats = result["image_stats"]
    if not isinstance(image_stats, dict) or not {"mode", "size", "pixel_min", "pixel_max"}.issubset(image_stats):
        raise ValueError("Inference result key 'image_stats' is missing required image details")
    if image_stats["mode"] != "L" or tuple(image_stats["size"]) != (frame.shape[1], frame.shape[0]):
        raise ValueError("Inference result 'image_stats' do not match the generated grayscale frame")
    if (image_stats["pixel_min"], image_stats["pixel_max"]) != (int(frame.min()), int(frame.max())):
        raise ValueError("Inference result 'image_stats' pixel range does not match 'generated_frame'")
    model_input = result["model_input"]
    if not isinstance(model_input, torch.Tensor) or model_input.ndim != 4 or model_input.shape[1] != 7:
        raise ValueError("Inference result key 'model_input' must be a BCHW tensor with seven trained input channels")
    return result


def load_trained_model(checkpoint_path: str | Path, device: torch.device | str):
    """Build the configured architecture and strictly load the saved weights."""
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint was not found: {path}")
    device = torch.device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = create_model(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Checkpoint is incompatible with {model.architecture}: {exc}") from exc
    model.eval()
    return model, checkpoint


def predict_tensor(model: torch.nn.Module, model_input: torch.Tensor, device=None):
    """Run the shared model forward pass and return CPU prediction plus elapsed seconds."""
    if model_input.ndim == 3:
        model_input = model_input.unsqueeze(0)
    if model_input.ndim != 4:
        raise ValueError(f"Expected a BCHW input tensor, received shape {tuple(model_input.shape)}")
    expected_channels = getattr(model, "input_channels", None)
    if expected_channels is not None and model_input.shape[1] != expected_channels:
        raise ValueError(f"Model expects {expected_channels} channels; received {model_input.shape[1]}")
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model_input = model_input.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = model(model_input)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(f"Expected one grayscale output channel, received shape {tuple(prediction.shape)}")
    if not torch.isfinite(prediction).all():
        raise ValueError("Model produced non-finite output values")
    return prediction.detach().cpu(), elapsed


def predict_model_input(model: torch.nn.Module, model_input: torch.Tensor, device=None) -> dict[str, Any]:
    """Predict from the trained seven-channel input and return the shared result dictionary."""
    prediction, inference_seconds = predict_tensor(model, model_input, device)
    output = tensor_to_display_image(prediction)
    flat = prediction.detach().float().reshape(-1)
    bounds = ((0.00, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.00))
    value_bins_pct = {
        f"{low:.2f}-{high:.2f}": float((((flat >= low) & (flat < high)) if high < 1.0 else ((flat >= low) & (flat <= high))).float().mean() * 100.0)
        for low, high in bounds
    }
    result = {
        "prediction": prediction,
        "generated_frame": output,
        "inference_seconds": inference_seconds,
        "prediction_stats": {
            "shape": tuple(prediction.shape),
            "dtype": str(prediction.dtype),
            "min": float(prediction.min()),
            "max": float(prediction.max()),
            "mean": float(prediction.mean()),
            "std": float(prediction.std(unbiased=False)),
            "value_bins_pct": value_bins_pct,
        },
        "image_stats": {
            "mode": "L",
            "size": (int(output.shape[1]), int(output.shape[0])),
            "pixel_min": int(output.min()),
            "pixel_max": int(output.max()),
        },
        "model_input": model_input.detach().cpu(),
    }
    return validate_inference_result(result)


def predict_image_pair(model: torch.nn.Module, frame_a: Image.Image, frame_b: Image.Image, device=None) -> dict[str, Any]:
    """Preprocess endpoint images, infer, restore aspect, and return the shared dictionary."""
    line_a, vertices_a, graph_a, _, content_box_a = image_to_inputs(frame_a)
    line_b, vertices_b, graph_b, _, _ = image_to_inputs(frame_b)
    model_input = torch.cat([
        line_a, vertices_a, graph_a,
        line_b, vertices_b, graph_b,
        torch.full_like(line_a, 0.5),
    ], dim=1)
    result = predict_model_input(model, model_input, device)
    result["geometry_mode"] = "Raster-guided inference (estimated skeleton geometry)"
    output = restore_aspect(result["generated_frame"], frame_a.size, content_box_a)
    result["generated_frame"] = output
    result["image_stats"] = {
        "mode": "L",
        "size": (int(output.shape[1]), int(output.shape[0])),
        "pixel_min": int(output.min()),
        "pixel_max": int(output.max()),
    }
    return validate_inference_result(result)

