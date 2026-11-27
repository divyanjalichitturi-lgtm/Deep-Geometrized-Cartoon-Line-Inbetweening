import json
import re
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


def extract_edges(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None")
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    return edges.astype(np.float32) / 255.0


def extract_contours(image: np.ndarray):
    edges = (extract_edges(image) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _numeric_pair_list(obj: Any):
    if not isinstance(obj, list) or not obj:
        return None
    out = []
    for item in obj:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                vals = [float(item[0]), float(item[1])]
            except Exception:
                return None
            out.append(vals)
        elif isinstance(item, dict):
            x = item.get("x", item.get("X"))
            y = item.get("y", item.get("Y"))
            if x is None or y is None:
                return None
            try:
                out.append([float(x), float(y)])
            except Exception:
                return None
        else:
            return None
    return out


def _int_pair_list(obj: Any):
    if not isinstance(obj, list) or not obj:
        return None
    out = []
    for item in obj:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return None
        try:
            a, b = int(item[0]), int(item[1])
        except Exception:
            return None
        out.append([a, b])
    return out


def _find_by_keys(obj: Any, keys):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in keys:
                yield v
            yield from _find_by_keys(v, keys)
    elif isinstance(obj, list):
        for v in obj:
            yield from _find_by_keys(v, keys)


def _fallback_vertices(obj: Any):
    candidates = []
    if isinstance(obj, dict):
        for v in obj.values():
            r = _fallback_vertices(v)
            if r:
                candidates.append(r)
    elif isinstance(obj, list):
        r = _numeric_pair_list(obj)
        if r and len(r) >= 2:
            return r
        for v in obj:
            r = _fallback_vertices(v)
            if r:
                candidates.append(r)
    return max(candidates, key=len) if candidates else None


def _fallback_connections(obj: Any, n_vertices: int):
    candidates = []
    if isinstance(obj, dict):
        for v in obj.values():
            r = _fallback_connections(v, n_vertices)
            if r:
                candidates.append(r)
    elif isinstance(obj, list):
        r = _int_pair_list(obj)
        if r:
            good = [e for e in r if 0 <= e[0] < n_vertices and 0 <= e[1] < n_vertices and e[0] != e[1]]
            if len(good) >= 1:
                return good
        for v in obj:
            r = _fallback_connections(v, n_vertices)
            if r:
                candidates.append(r)
    return max(candidates, key=len) if candidates else None


def load_geometry_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vertex_keys = {"vertices", "vertex", "vtx", "points", "vertexlocations", "vertexlocation", "locations", "vertices2d", "vertex_pos", "vertex_positions"}
    connection_keys = {"connections", "connection", "edges", "adjacency", "adjacent", "topology", "links", "connections_table"}

    vertices = None
    for candidate in _find_by_keys(data, vertex_keys):
        vertices = _numeric_pair_list(candidate)
        if vertices and len(vertices) >= 2:
            break
    if vertices is None:
        vertices = _fallback_vertices(data)

    connections = None
    if vertices:
        for candidate in _find_by_keys(data, connection_keys):
            connections = _int_pair_list(candidate)
            if connections:
                connections = [e for e in connections if 0 <= e[0] < len(vertices) and 0 <= e[1] < len(vertices) and e[0] != e[1]]
                if connections:
                    break
    if connections is None and vertices:
        connections = _fallback_connections(data, len(vertices)) or []

    if not vertices:
        raise ValueError(f"Could not find 2D vertex coordinates in {path}")

    return np.asarray(vertices, dtype=np.float32), connections


def _scale_vertices(vertices: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int):
    v = vertices.copy()
    # ML240-style labels are normally pixel coordinates. If values look normalized, convert first.
    if np.nanmax(v[:, 0]) <= 1.5 and np.nanmax(v[:, 1]) <= 1.5:
        v[:, 0] *= src_w
        v[:, 1] *= src_h
    v[:, 0] *= dst_w / float(src_w)
    v[:, 1] *= dst_h / float(src_h)
    return v


def geometry_to_maps(label_path: str, src_size: Tuple[int, int], dst_size: Tuple[int, int] = (256, 256)):
    vertices, connections = load_geometry_json(label_path)
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    vertices = _scale_vertices(vertices, src_w, src_h, dst_w, dst_h)

    vertex_map = np.zeros((dst_h, dst_w), dtype=np.float32)
    edge_map = np.zeros((dst_h, dst_w), dtype=np.float32)

    for x, y in vertices:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < dst_w and 0 <= yi < dst_h:
            cv2.circle(vertex_map, (xi, yi), 2, 1.0, -1)

    for a, b in connections:
        x1, y1 = vertices[a]
        x2, y2 = vertices[b]
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.line(edge_map, p1, p2, 1.0, 1)

    return vertex_map, edge_map, vertices, connections


def image_geometry_maps(image: np.ndarray, dst_size=(256, 256)):
    """Approximate ML240 graph maps from raster stroke centerlines.

    ML240 maps represent annotated graph vertices (radius-2 disks) and graph
    connections (one-pixel lines), not the two-sided boundaries returned by
    Canny. For uploaded line art, Otsu binarization followed by skeletonization
    estimates those centerlines. Deterministic samples at roughly three-pixel
    intervals provide vertex coverage comparable to the dense ML240 polylines;
    endpoints and junctions are always included. This remains an estimate, not
    a substitute for the source annotation.
    """
    if image is None:
        raise ValueError("Input image is None")
    dst_w, dst_h = dst_size
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Geometry input must be a grayscale or RGB image")
    if gray.shape[::-1] != (dst_w, dst_h):
        gray = cv2.resize(gray, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
    gray = np.asarray(gray, dtype=np.uint8)

    # A dark-foreground mask is needed only to estimate geometry. The model's
    # raster input remains the original continuous 1 - grayscale/255 map.
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    threshold = int(np.clip(threshold, 32, 224))
    foreground = gray <= threshold
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    cleaned = np.zeros_like(foreground)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= 2:
            cleaned[labels == label] = True
    skeleton = skeletonize(cleaned)
    graph_map = skeleton.astype(np.float32)
    vertex_map = np.zeros((dst_h, dst_w), dtype=np.float32)
    points = np.argwhere(skeleton)
    if len(points):
        # Sorted row-major samples approximate the dense ML240 polyline nodes;
        # include topology-critical points even when they fall between samples.
        selected = set(map(tuple, points[::3]))
        padded = np.pad(skeleton.astype(np.uint8), 1)
        neighbor_count = np.zeros_like(skeleton, dtype=np.uint8)
        for dy in range(3):
            for dx in range(3):
                if dx == 1 and dy == 1:
                    continue
                neighbor_count += padded[dy:dy + dst_h, dx:dx + dst_w]
        critical = np.argwhere(skeleton & ((neighbor_count == 1) | (neighbor_count >= 3)))
        selected.update(map(tuple, critical))
        for y, x in sorted(selected):
            cv2.circle(vertex_map, (int(x), int(y)), 2, 1.0, -1)
    return vertex_map, graph_map


def save_geometry_overlay(image, vertices, connections, output_path):
    canvas = image.copy()
    for a, b in connections:
        p1 = tuple(np.round(vertices[a]).astype(int))
        p2 = tuple(np.round(vertices[b]).astype(int))
        cv2.line(canvas, p1, p2, (255, 0, 0), 1)
    for x, y in vertices:
        cv2.circle(canvas, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
