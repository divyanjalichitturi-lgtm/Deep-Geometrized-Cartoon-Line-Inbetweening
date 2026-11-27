import re
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .geometry import geometry_to_maps, extract_edges

IMAGE_SIZE = (256, 256)
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def _frame_number(path):
    m = re.findall(r"(\d+)", path.stem)
    return int(m[-1]) if m else -1


def _label_for(frame_path, label_dir):
    n = _frame_number(frame_path)
    # Match the complete frame index so e.g. frame 1 cannot select label 10.
    candidates = [p for p in label_dir.glob("*.json") if _frame_number(p) == n]
    return sorted(set(candidates))[0] if candidates else None


class ML240InbetweenDataset(Dataset):
    def __init__(self, root_dir, image_size=IMAGE_SIZE):
        self.root = Path(root_dir)
        self.image_size = image_size
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.samples = []
        for seq in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            frames_dir = seq / "frames"
            labels_dir = seq / "labels"
            if not frames_dir.exists() or not labels_dir.exists():
                continue
            frames = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS], key=_frame_number)
            for i in range(len(frames) - 2):
                a, mid, b = frames[i], frames[i+1], frames[i+2]
                la, lm, lb = _label_for(a, labels_dir), _label_for(mid, labels_dir), _label_for(b, labels_dir)
                if la and lm and lb:
                    self.samples.append((a, la, mid, lm, b, lb))
        if not self.samples:
            raise RuntimeError(f"No valid ML240 triples found under {self.root}")
        print(f"Samples in {self.root}: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _load(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read {path}")
        src_h, src_w = img.shape[:2]
        # INTER_AREA is used for downsampling to retain average line coverage;
        # upsampling uses cubic interpolation to avoid blocky contours.
        interpolation = cv2.INTER_AREA if src_w > self.image_size[0] or src_h > self.image_size[1] else cv2.INTER_CUBIC
        img = cv2.resize(img, self.image_size, interpolation=interpolation)
        line = 1.0 - img.astype(np.float32) / 255.0
        return line, (src_w, src_h)

    def _item(self, frame_path, label_path):
        line, src_size = self._load(frame_path)
        # Geometry maps are generated from original-coordinate labels and transformed to 256x256.
        vertex_map, graph_map, _, _ = geometry_to_maps(str(label_path), src_size, self.image_size)
        return line, vertex_map, graph_map

    def __getitem__(self, idx):
        a, la, mid, lm, b, lb = self.samples[idx]
        line_a, va, ga = self._item(a, la)
        line_m, vm, gm = self._item(mid, lm)
        line_b, vb, gb = self._item(b, lb)
        x = np.stack([line_a, va, ga, line_b, vb, gb, np.full(self.image_size[::-1], 0.5, dtype=np.float32)])
        target = line_m[None, ...].astype(np.float32)
        return {"input": torch.from_numpy(x).float(), "target": torch.from_numpy(target).float(), "geometry_target": torch.from_numpy(np.stack([vm, gm])).float()}
