# DeepLine AI — Geometry-Aware Cartoon Line Inbetweening

A geometry-aware deep learning project that predicts an intermediate cartoon line-art frame from two endpoint drawings. The implementation uses ML240 geometry information and is inspired by AnimeInbet (ICCV 2023).

> **Note:** This is a practical student-scale implementation inspired by AnimeInbet. It is not an exact reproduction of the original ICCV 2023 architecture.

## 1. Project Overview

Cartoon animation requires intermediate drawings between two keyframes to create smooth motion. Manually drawing these intermediate frames can be time-consuming.

DeepLine AI aims to automate this process using a geometry-aware deep learning model.

The application accepts two endpoint cartoon drawings and predicts a candidate middle frame using raster and geometry features.

### Objectives

- Predict an intermediate cartoon line-art frame from two endpoint drawings.
- Incorporate vertex and graph-edge information into the learning process.
- Train using ML240 geometry annotations.
- Evaluate predictions against held-out ground-truth frames.
- Provide an interactive Streamlit application for inference and visualization.

## 2. Research Inspiration and Dataset

This project is inspired by:

**Deep Geometrized Cartoon Line Inbetweening (AnimeInbet), ICCV 2023.**

The implementation uses a reduced subset of ML240 / MixamoLine240.

### Dataset configuration

| Split | Number of sequences |
|---|---:|
| Training | 10 |
| Validation | 2 |
| Testing | 3 |

The dataset is divided at the sequence level to prevent frames from the same source sequence from appearing in multiple splits.

Training uses ML240 vertex and connection JSON annotations to construct geometry features.

**Important dataset note:** The prepared dataset is not included in this GitHub repository. You must prepare it locally using the provided preparation script and the ML240 source files.

## 3. Model Architecture

The project uses a geometry-aware deep learning network designed to combine raster information and geometric structure.

The model uses two feature streams:

### Raster stream
- Starting line-art frame.
- Ending line-art frame.
- Interpolation time.

### Geometry stream
- Endpoint vertex heatmaps.
- Graph-edge maps.

The two streams are combined through gated feature fusion. Residual blocks, channel attention, skip connections, and a decoder are used to reconstruct the intermediate line-art image.

The model produces a 256 × 256 grayscale intensity map.

The implementation uses geometry features during training and estimates geometry from uploaded raster images during application inference.

## 4. Training and Loss Functions

Training combines multiple objectives to improve reconstruction quality and structural consistency.

The loss components include:

- L1 reconstruction loss.
- Weighted Mean Squared Error (MSE).
- Finite-difference edge loss.
- Differentiable local SSIM-style structural loss.

Loss weights are configured in `config.py`.

The SSIM-style term can be disabled using:

```bash
--ssim-weight 0
```

Training uses AdamW optimization, reproducible random seeds, validation-based learning-rate reduction, checkpoint saving, and early stopping.

## 5. Project Features

- Geometry-aware cartoon frame prediction.
- ML240 vertex and graph-edge feature processing.
- Shared inference functions for testing and application prediction.
- Strict checkpoint loading.
- Image preprocessing with aspect-ratio preservation.
- Raw prediction diagnostics.
- Held-out evaluation metrics.
- Interactive Streamlit interface.

### Streamlit application pages

1. Home
2. Inbetweening Studio
3. How It Works
4. Geometry Analyzer
5. Model & Performance
6. About

Users can upload endpoint images, preview the inputs, generate an intermediate frame, inspect diagnostics, and download the generated PNG.

Supported image formats include PNG, JPG, JPEG, BMP, and WebP when supported by the installed Pillow version.

## 6. Technologies Used

- Python
- PyTorch
- NumPy
- OpenCV
- Pillow
- Streamlit
- Matplotlib
- Scikit-image

## 7. Installation

Use Python 3.12 where available.

Clone the repository:

```bash
git clone https://github.com/divyanjalichitturi-lgtm/Deep-Geometrized-Cartoon-Line-Inbetweening.git
```

Move into the project directory:

```bash
cd Deep-Geometrized-Cartoon-Line-Inbetweening
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for your CUDA configuration if GPU acceleration is required. CPU execution is also supported.

## 8. Dataset Preparation

The dataset is excluded from this GitHub repository.

Prepare the reduced ML240 dataset locally using:

```bash
python prepare_ml240.py --source "C:\path\to\ml240data" --output dataset --train 10 --val 2 --test 3 --size 256 --seed 42
```

Replace the source path with the location of your extracted ML240 dataset.

Validate the prepared dataset:

```bash
python validate_dataset.py
```

The preparation script selects complete sequences and creates training, validation, and testing splits.

## 9. Training

After preparing the dataset, run a short training experiment:

```bash
python train.py --epochs 10 --batch-size 2
```

For a longer experiment:

```bash
python train.py --epochs 30 --batch-size 4 --max-train-samples 1000 --max-val-samples 200
```

Training checkpoints, history files, and loss curves are generated locally.

**Note:** A trained model checkpoint is not included in this GitHub repository. You must train the model locally or obtain a compatible checkpoint before running checkpoint-based inference and evaluation.

## 10. Testing and Evaluation

Run the evaluation script:

```bash
python test.py
```

To evaluate another test sample:

```bash
python test.py --index 1
```

The project evaluates predictions using:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Chamfer Distance

The metrics compare the generated frame against the corresponding ground-truth middle frame.

Chamfer distance is not evaluated when either binary line map is empty.

## 11. Run the Streamlit Application

Start the application using:

```bash
streamlit run app.py
```

The application provides an interface to upload two endpoint drawings and generate a candidate intermediate frame.

Users can:
- Upload supported image formats.
- Preview the endpoint images.
- Generate an intermediate line-art frame.
- Download the generated PNG.
- Inspect estimated geometry and prediction diagnostics.
- View held-out evaluation information.

Uploaded-image metrics are not reported because the application does not receive ground-truth middle frames for user-provided images.

## 12. Inference Diagnostics

The project includes a diagnostic script:

```bash
python diagnose_inference.py --index 0
```

It saves raw prediction information, a display PNG, and a comparison image for inspection.

The testing script and Streamlit application use shared inference functions to maintain consistent model loading and prediction behavior.

## 13. Existing Sample Evaluation

The latest local evaluation report for test sample 0 records:

| Metric | Result |
|---|---:|
| MAE | 0.01072 |
| MSE | 0.003007 |
| PSNR | 25.22 dB |
| SSIM | 0.9632 |
| Chamfer Distance | 0.45966 |

These values correspond to one held-out test sample using the checkpoint from epoch 10. They are not guarantees of performance on other samples or user-uploaded images.

## 14. Project Structure

```text
Deep-Geometrized-Cartoon-Line-Inbetweening/
│
├── app.py
├── train.py
├── test.py
├── config.py
├── project_check.py
├── diagnose_inference.py
├── prepare_ml240.py
├── validate_dataset.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
└── src/
    ├── __init__.py
    ├── dataset.py
    ├── geometry.py
    ├── inference.py
    ├── losses.py
    ├── metrics.py
    ├── model.py
    └── preprocess.py
```

**Note:** The dataset, trained checkpoints, and generated output files are stored locally and are excluded from the public repository. The `.gitkeep` files preserve the empty directory structure.

## 15. Project Validation

Run the project validation script:

```bash
python project_check.py
```

It checks project files, package imports, dataset structure, Python syntax, and checkpoint compatibility.

A compatible trained checkpoint and prepared dataset are required for the corresponding training, testing, and inference workflows.

## 16. Limitations and Future Improvements

### Current limitations

- The training dataset is a small subset of ML240.
- The model uses a fixed 256 × 256 training canvas.
- Geometry estimated from uploaded raster images is approximate.
- Restoring an output to a larger resolution cannot recover details that were not learned at the model's training resolution.
- The implementation does not reproduce the full correspondence reasoning of the original AnimeInbet method.
- Evaluation on one sample does not establish broad generalization.

### Future improvements

- Increase training dataset coverage.
- Improve geometry estimation for uploaded drawings.
- Explore correspondence-aware supervision.
- Evaluate using multiple random seeds.
- Test on different cartoon drawing styles.
- Improve fine-detail preservation in generated frames.

## 17. License and Attribution

This project is an independent student-scale implementation inspired by the AnimeInbet research paper.

Please check the ML240 / MixamoLine240 dataset terms before redistributing or publishing dataset content.

The repository does not grant permission to redistribute the original dataset or research materials.

Cite the original AnimeInbet paper and dataset in academic reports when using their work or data.

---

## Author

**DeepLine AI — Geometry-Aware Cartoon Line Inbetweening**

GitHub Repository:

https://github.com/divyanjalichitturi-lgtm/Deep-Geometrized-Cartoon-Line-Inbetweening
