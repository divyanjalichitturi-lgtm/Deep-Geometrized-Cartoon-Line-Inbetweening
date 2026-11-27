# DeepLine AI — Geometry-Aware Cartoon Line Inbetweening

An ML240-based geometry-aware deep learning implementation inspired by AnimeInbet. The project predicts a candidate middle line-art frame from two endpoint drawings and uses geometry maps during learning and inference.

> This is a practical student-scale implementation. It is **not an exact reproduction** of the ICCV 2023 AnimeInbet architecture.

## Problem and objectives

Cartoon animation requires drawings between key poses. This project learns to reconstruct the middle frame from the preceding and following raster frames while incorporating vertex and graph-edge cues. Its objectives are to preserve the existing ML240 preparation workflow, train on real geometry labels, evaluate against held-out frames, and expose inference and diagnostics through a research-oriented application.

## Research inspiration and dataset

The work is inspired by *Deep Geometrized Cartoon Line Inbetweening* (AnimeInbet, ICCV 2023). The original method contains correspondence and geometry reasoning beyond this student implementation.

Training uses the prepared reduced subset of ML240 / MixamoLine240: **10 training, 2 validation, and 3 testing sequences**. Splits are at sequence level; adjacent frames within a sequence form triples, and a source sequence is assigned to one split only. Training reads the ML240 vertex and connection JSON labels. At upload inference, geometry is estimated from the raster's contours and explicitly marked as estimated rather than ground truth.

The existing prepared `dataset/` is retained. It does not need to be recreated to use this version.

## Architecture and pipeline

The verified checkpoint uses the `GeometryAwareDualBranchNet` (Geometry-Aware Dual-Branch Residual U-Net), with two streams:

- Raster stream: start line map, end line map, and interpolation time.
- Geometry stream: endpoint vertex heatmaps and graph-edge maps.
- Residual blocks use channel attention. Gated fusion combines both streams at multiple scales; skip connections feed a decoder and one-channel line-intensity reconstruction head.

The model emits a 256 × 256 grayscale intensity map, where larger values represent dark line pixels. Training combines L1 reconstruction, weighted MSE, finite-difference edge loss, and a differentiable local SSIM-style structural loss. Weights are set in `config.py`; the SSIM term may be disabled with `--ssim-weight 0`.

Preprocessing preserves aspect ratio for uploaded images using white letterboxing. Dataset frames are resized to the fixed training canvas. The model output is cropped from the letterbox and restored to the start frame's original resolution. The output conversion maps the trained sigmoid [0, 1] line-strength convention directly to grayscale (high means dark), with no percentile stretch, gamma adjustment, threshold, or added geometry. Reported metrics are calculated on the untouched raw prediction. Run `python diagnose_inference.py --index 0` to save raw prediction values, the display PNG, and a comparison using the same estimated-geometry path as Streamlit.

Checkpoint loading and forward passes are centralized in `src/inference.py`. Both `test.py` and Streamlit call the same strict loader and shared prediction functions, which return the same validated dictionary contract. Uploaded-pair prediction returns `prediction`, `generated_frame`, `inference_seconds`, `prediction_stats`, `image_stats`, and `model_input`. The upload path uses the shared project preprocessor to estimate inference-time geometry, then assembles the same seven-channel model input contract used by training.

## Installation

Use Python 3.12 where available. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use a PyTorch wheel appropriate for your CUDA setup if GPU training is desired; otherwise the CPU build is supported. The code selects CUDA automatically when available.

### Quick start (PowerShell)

```powershell
cd "C:\Users\Divya Chitturi\OneDrive\Desktop\Deep_Geometrized_Cartoon_Line_Inbetweening\Deep_Geometrized_Cartoon_Line_Inbetweening"
python -m pip install -r requirements.txt
python project_check.py
python test.py --index 0
streamlit run app.py
```

## Dataset preparation

The prepared dataset is already included in this project. If it must be regenerated from the source archive, run this command from the project directory (edit the source path to the local extracted ML240 folder):

```powershell
python prepare_ml240.py --source "C:\Users\Divya Chitturi\OneDrive\Desktop\ML240_TEMP\ml240data" --output dataset --train 10 --val 2 --test 3 --size 256 --seed 42
```

Preparation selects complete sequences without replacement and writes normalized frame/geometry pairs. Existing split data is protected; use a new `--output` directory when preparing another copy. Check the resulting structure with:

```powershell
python validate_dataset.py
```

## Training

Project paths are anchored to each script's location, so commands may be run from another current directory by passing the script's path. From the project directory, a short run is:

```powershell
python train.py --epochs 10 --batch-size 2
```

For a longer run or capped sample counts:

```powershell
python train.py --epochs 30 --batch-size 4 --max-train-samples 1000 --max-val-samples 200
```

Continue from a full training checkpoint:

```powershell
python train.py --epochs 30 --batch-size 2 --resume outputs/checkpoints/run_<timestamp>/epoch_005.pth
```

Training is seeded, uses AdamW and validation-driven learning-rate reduction, saves per-epoch checkpoints, tracks CSV history and a PNG loss curve, and stops early when validation loss stalls. `models/best_model.pth` contains model, optimizer, scheduler, epoch, validation loss, configuration, and history. Incompatible previous weights are archived under `outputs/checkpoints/` before starting a fresh run.

## Testing and evaluation

Evaluate held-out test sample 0:

```powershell
python test.py
```

Choose another sample with `python test.py --index 1`. The script reports MAE, MSE, PSNR, SSIM, and binary-line Chamfer distance calculated against the selected ground-truth middle frame. It writes generated and reference PNGs, a four-panel comparison, and per-run plus latest JSON/CSV metrics under `outputs/`. Chamfer distance is `Not evaluated` when either binary line map is empty.

## Streamlit application

```powershell
streamlit run app.py
```

DeepLine AI includes Home, Inbetweening Studio, How It Works, Geometry Analyzer, Model & Performance, and About pages. Upload PNG, JPG, JPEG, BMP, or WebP endpoint frames (WebP is offered when Pillow reports WebP support), review previews, generate and download the actual grayscale model output as PNG, inspect raw prediction/PNG diagnostics and estimated geometry, and view held-out evaluation. Each generation is saved to `outputs/generated/` with its comparison in `outputs/comparisons/`. Uploaded-image metrics are not reported because no ground truth is supplied. Home navigation and Studio reset use callbacks/key rotation to avoid mutating widget state after creation; stale session results from older schemas are discarded.

## Existing sample result

The latest local evaluation report for test sample 0 records MAE **0.01072**, MSE **0.003007**, PSNR **25.22 dB**, SSIM **0.9632**, and Chamfer distance **0.45966** (checkpoint epoch 10). These values describe that held-out sample and are not guarantees for user uploads; inference time is recorded per run and varies with CPU load.

## Project checks

```powershell
python project_check.py
```

The check reports file/directory presence, package imports, dataset splits, Python syntax, and whether the existing checkpoint strictly matches the model architecture. The project does not silently bypass checkpoint mismatches.

## Outputs

```text
models/best_model.pth
outputs/
  checkpoints/
  generated/
  comparisons/
  metrics/
  training_history.csv
  training_curves.png
```

## Project structure

```text
app.py                    Streamlit application
train.py / test.py        Training and held-out evaluation
config.py                 Model, data and device settings
project_check.py          Automated project validation
src/model.py              Trained network architecture
src/inference.py          Shared strict loading and prediction functions
src/dataset.py            Sequence-level ML240 triple loader
src/preprocess.py         Raster/geometry features and output conversion
src/geometry.py           ML240 geometry parsing and inference estimates
src/losses.py / metrics.py Training losses and evaluation metrics
dataset/                  Existing reduced sequence split
models/best_model.pth     Existing trained checkpoint
outputs/                  Generated frames, comparisons and reports
```

## Limitations and future work

The prepared subset is small and does not establish broad generalization. The output canvas is 256 × 256 during inference; restoration to larger dimensions cannot recover detail absent from the model. Image-based inference geometry is approximate, and this system does not implement AnimeInbet's full correspondence reasoning. Future work includes broader sequence coverage, correspondence supervision, multiple-seed evaluation, and validation on distinct drawing styles.

## License considerations

Confirm ML240 / MixamoLine240 terms before redistribution or publication. Cite the dataset and AnimeInbet research paper in reports. This repository does not grant rights to redistribute either resource.



