"""Raster preprocessing shared by Streamlit inference and geometry inspection."""
import cv2
import numpy as np
import torch
from PIL import Image

from .geometry import extract_edges, image_geometry_maps

IMAGE_SIZE = (256, 256)


def load_rgb(source):
    if isinstance(source, Image.Image): return np.asarray(source.convert("RGB"))
    if isinstance(source, np.ndarray):
        if source.ndim == 2: return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        if source.ndim == 3 and source.shape[2] >= 3: return source[:, :, :3].astype(np.uint8)
        raise ValueError("Image must be grayscale or RGB/RGBA")
    raise TypeError("Unsupported image source")


def letterbox_rgb(rgb, size=IMAGE_SIZE):
    """Fit without changing aspect ratio; pad with white so lines stay legible."""
    dst_w, dst_h = size; src_h, src_w = rgb.shape[:2]
    if src_w < 1 or src_h < 1: raise ValueError("Image has empty dimensions")
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
    x0, y0 = (dst_w - new_w) // 2, (dst_h - new_h) // 2
    canvas = np.full((dst_h, dst_w, 3), 255, dtype=np.uint8)
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas, (x0, y0, new_w, new_h)


def image_to_inputs(source, size=IMAGE_SIZE):
    rgb = load_rgb(source)
    canvas, content_box = letterbox_rgb(rgb, size)
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # Training convention: line darkness is represented by high values.
    line = 1.0 - gray
    vertex_map, graph_edge_map = image_geometry_maps(canvas, size)
    tensors = [torch.from_numpy(x.copy()).float()[None, None] for x in (line, vertex_map, graph_edge_map)]
    return tensors[0], tensors[1], tensors[2], canvas, content_box


def restore_aspect(image, original_size, content_box):
    """Remove inference padding and resize output to (width, height) input size."""
    x0, y0, width, height = content_box
    cropped = image[y0:y0 + height, x0:x0 + width]
    dst_w, dst_h = original_size
    method = cv2.INTER_AREA if dst_w < width or dst_h < height else cv2.INTER_NEAREST
    return cv2.resize(cropped, (dst_w, dst_h), interpolation=method)


def tensor_to_display_image(tensor):
    """Map the trained dark-line intensity [0, 1] to conventional grayscale.

    Training uses ``1 - grayscale / 255`` for both inputs and targets, so the
    model's sigmoid output already is line darkness. Convert it directly to
    uint8 grayscale: no percentile stretch, gamma, or threshold is applied.
    """
    x = tensor.detach().cpu().squeeze().float().numpy()
    if x.ndim != 2:
        raise ValueError(f"Expected one grayscale prediction, received shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Prediction contains non-finite values")
    # Sigmoid is the trained head activation; clipping only protects uint8
    # conversion from tiny numerical excursions beyond its [0, 1] range.
    return np.rint(255.0 * (1.0 - np.clip(x, 0.0, 1.0))).astype(np.uint8)


def tensor_to_image(tensor):
    """Backward-compatible alias for the single shared PNG conversion."""
    return tensor_to_display_image(tensor)

