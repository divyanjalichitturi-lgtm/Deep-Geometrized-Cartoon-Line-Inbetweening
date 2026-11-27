"""Report PASS/FAIL checks for source files, data, dependencies and checkpoint."""
from pathlib import Path
import importlib
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = ["app.py", "train.py", "test.py", "diagnose_inference.py", "prepare_ml240.py", "validate_dataset.py", "config.py", "src/model.py", "src/dataset.py", "src/geometry.py", "src/losses.py", "src/metrics.py", "src/preprocess.py", "src/inference.py", "requirements.txt", "README.md"]
REQUIRED_IMPORTS = ["torch", "cv2", "numpy", "PIL", "skimage", "streamlit", "src.inference"]


def report(name, passed, details=""):
    print(f"{'PASS' if passed else 'FAIL'} | {name}" + (f" | {details}" if details else ""))
    return bool(passed)


def main():
    checks = []
    missing = [name for name in REQUIRED_FILES if not (PROJECT_ROOT / name).is_file()]
    checks.append(report("required files", not missing, ", ".join(missing) if missing else f"{len(REQUIRED_FILES)} present"))
    dirs = ["dataset/train", "dataset/val", "dataset/test", "models", "outputs/generated", "outputs/comparisons", "outputs/metrics", "outputs/checkpoints"]
    absent = [name for name in dirs if not (PROJECT_ROOT / name).is_dir()]
    checks.append(report("required directories", not absent, ", ".join(absent) if absent else f"{len(dirs)} present"))
    imports_ok = True
    for module in REQUIRED_IMPORTS:
        try: importlib.import_module(module)
        except Exception as exc: imports_ok = False; print(f"FAIL | import {module} | {exc}")
    if imports_ok: print(f"PASS | Python imports | {', '.join(REQUIRED_IMPORTS)}")
    checks.append(imports_ok)
    structure_ok = True; seqs = {}
    for split in ("train", "val", "test"):
        folder = PROJECT_ROOT / "dataset" / split
        seqs[split] = [p for p in folder.iterdir() if p.is_dir()] if folder.is_dir() else []
        if not seqs[split]: structure_ok = False
    checks.append(report("dataset split structure", structure_ok, json.dumps({k: len(v) for k, v in seqs.items()})))
    model_path = PROJECT_ROOT / "models" / "best_model.pth"
    checkpoint_ok = False; detail = "checkpoint missing"
    try:
        import torch
        from src.model import create_model
        if model_path.is_file():
            payload = torch.load(model_path, map_location="cpu", weights_only=False)
            state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
            create_model().load_state_dict(state, strict=True)
            checkpoint_ok, detail = True, "strict state-dict load succeeded"
    except Exception as exc: detail = str(exc)
    checks.append(report("checkpoint compatibility", checkpoint_ok, detail))
    syntax_ok = True
    for path in PROJECT_ROOT.rglob("*.py"):
        if ".venv" in path.parts or "__pycache__" in path.parts: continue
        try: compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except Exception as exc: syntax_ok = False; print(f"FAIL | syntax {path.relative_to(PROJECT_ROOT)} | {exc}")
    if syntax_ok: print("PASS | Python syntax | all project .py files")
    checks.append(syntax_ok)
    print(f"\nOverall: {'PASS' if all(checks) else 'FAIL'} ({sum(checks)}/{len(checks)} checks)")
    return 0 if all(checks) else 1


if __name__ == "__main__": sys.exit(main())
