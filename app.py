"""DeepLine AI: an ML240-based geometry-aware line-inbetweening studio."""
from pathlib import Path
import hashlib
import inspect
import io
import json
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError, features

from config import DEVICE, IMAGE_SIZE, LOSS_WEIGHTS, MODEL_PATH, PROJECT_ROOT
from src.geometry import extract_edges
from src.inference import load_trained_model, predict_image_pair, validate_inference_result
from src.preprocess import image_to_inputs

st.set_page_config(page_title="DeepLine AI", page_icon="✦", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: radial-gradient(ellipse at 15% 0%, #17263b 0, #0b111a 45%, #080d13 100%); color:#edf3fa; }
[data-testid="stSidebar"] { background:#0c141f; border-right:1px solid #243448; }
[data-testid="stSidebar"] * { color:#dbe6f3; }
.hero { padding:2.4rem 2.7rem; border:1px solid #263950; border-radius:24px; background:linear-gradient(120deg,rgba(35,68,101,.78),rgba(16,27,42,.78)); margin:1rem 0 1.6rem; }
.eyebrow { color:#79d8c2; letter-spacing:.16em; font-size:.76rem; font-weight:700; text-transform:uppercase; }
.hero h1 { font-size:3rem; margin:.5rem 0 .3rem; color:#f5f8fc; }
.hero p { color:#b8c8d9; font-size:1.08rem; max-width:850px; }
.card { background:#111c29; border:1px solid #26384c; border-radius:16px; padding:1.1rem 1.2rem; height:100%; }
.card h3 { color:#eaf2fb; margin:.1rem 0 .45rem; font-size:1.05rem; }
.card p { color:#9fb1c4; margin:0; font-size:.94rem; }
.step { padding:.85rem; text-align:center; border-radius:12px; background:#142235; border:1px solid #2a3b50; color:#c7d6e5; min-height:72px; }
.step b { display:block; color:#79d8c2; font-size:.72rem; letter-spacing:.1em; margin-bottom:.25rem; }
.subtle { color:#9fb1c4; }
div.stButton > button[kind="primary"] { background:linear-gradient(100deg,#50c9b5,#79aef8); color:#07131d; font-weight:800; border:0; border-radius:10px; min-height:3rem; }
div.stDownloadButton > button { border-radius:10px; font-weight:700; }
hr { border-color:#27384b; }
</style>
""", unsafe_allow_html=True)

UPLOAD_TYPES = ["png", "jpg", "jpeg", "bmp"] + (["webp"] if features.check("webp") else [])
IMAGE_STRETCH = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.image).parameters
    else {"use_container_width": True}
)


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, device_name: str, checkpoint_signature: str):
    path = Path(checkpoint_path)
    if not path.is_file(): return None, "No trained model found. Train first with `python train.py`."
    try:
        net, _ = load_trained_model(path, device_name)
        return net, None
    except Exception as exc:
        return None, f"The checkpoint could not be loaded with the current architecture: {exc}. Run a fresh training session."


def upload_image(upload):
    if upload is None: return None
    try:
        upload.seek(0)
        with Image.open(upload) as image:
            if image.width < 1 or image.height < 1:
                raise ValueError("The image has invalid dimensions.")
            if image.width * image.height > 50_000_000:
                raise ValueError("The image is larger than the supported 50 megapixel limit.")
            image.load()
            return image.convert("RGB").copy()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        st.error(f"This image could not be opened. Choose a valid supported image. Details: {exc}")
        return None


def navigate_to_studio():
    """Called before the next rerun, so the radio key is set before creation."""
    st.session_state["page"] = "INBETWEENING STUDIO"


def reset_studio():
    """Rotate uploader widget keys; Streamlit owns and clears their old values."""
    st.session_state["studio_upload_generation"] = int(st.session_state.get("studio_upload_generation", 0)) + 1
    st.session_state.pop("result", None)


def input_signature(upload_a, upload_b):
    if upload_a is None or upload_b is None:
        return None
    digest = hashlib.sha256()
    digest.update(upload_a.getvalue())
    digest.update(b"\0")
    digest.update(upload_b.getvalue())
    return digest.hexdigest()


def make_comparison(image_a, generated, image_b):
    """Build an aspect-preserving, labeled three-frame comparison."""
    tile_w, tile_h, label_h = 480, 390, 42
    comparison = Image.new("RGB", (tile_w * 3, tile_h + label_h), "#e9eef3")
    draw = ImageDraw.Draw(comparison)
    for index, (label, image) in enumerate((
        ("FRAME A", image_a),
        ("AI GENERATED FRAME", generated),
        ("FRAME B", image_b),
    )):
        x = index * tile_w
        draw.text((x + 14, 13), label, fill="#173047")
        contained = ImageOps.contain(image.convert("RGB"), (tile_w - 20, tile_h - 16))
        comparison.paste(contained, (x + (tile_w - contained.width) // 2, label_h + (tile_h - contained.height) // 2))
    return comparison


def heading(kicker, title, description=None):
    st.markdown(f'<div class="eyebrow">{kicker}</div><h1 style="margin:.2rem 0 .4rem;color:#f2f6fb">{title}</h1>', unsafe_allow_html=True)
    if description: st.markdown(f'<p class="subtle">{description}</p>', unsafe_allow_html=True)


def show_home():
    st.markdown('<div class="hero"><div class="eyebrow">Research Studio · ML240</div><h1>Deep Geometrized Cartoon Line Inbetweening</h1><div style="font-size:1.25rem;color:#b9d2e8">Geometry-aware AI for cartoon keyframes</div><p>Generate a plausible intermediate cartoon line-art frame from two keyframes with the project’s trained deep learning model.</p></div>', unsafe_allow_html=True)
    workflow = ["Frame A", "Geometry extraction", "Dual-branch network", "Generated middle", "Frame B"]
    cols = st.columns(len(workflow))
    for i, (col, label) in enumerate(zip(cols, workflow)):
        with col: st.markdown(f'<div class="step"><b>STEP {i+1}</b>{label}</div>', unsafe_allow_html=True)
    st.write("")
    cards = [("⌁", "Geometry-aware AI", "Raster cues fuse with estimated vertex and graph-edge maps."), ("▦", "ML240 dataset", "Training uses the prepared sequence-level ML240 subset and its geometry labels."), ("◈", "Deep learning", "A compact residual dual-branch encoder-decoder with gated multi-scale fusion."), ("✎", "Line-art reconstruction", "The model predicts a single-channel middle-frame line-intensity map.")]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, cards):
        with col: st.markdown(f'<div class="card"><h3>{icon} &nbsp;{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
    st.write("")
    st.button("Start Inbetweening  →", type="primary", on_click=navigate_to_studio)
    st.caption("Research positioning: an ML240-based geometry-aware deep learning implementation inspired by AnimeInbet; not an exact reproduction of the ICCV 2023 architecture.")


def show_studio(model, model_error):
    heading("CREATE", "Inbetweening Studio", "Upload two keyframes and synthesize one intermediate line-art frame.")
    st.info("Raster-guided inference · uploaded-image geometry is estimated from raster strokes, not read from ML240 annotations. Best results are obtained with clean cartoon line-art frames similar to the training dataset.")
    if model_error: st.warning(model_error)
    upload_generation = int(st.session_state.get("studio_upload_generation", 0))
    st.button("Reset Studio", on_click=reset_studio, key=f"reset_studio_{upload_generation}")
    a_col, b_col = st.columns(2)
    with a_col:
        st.markdown("### START FRAME")
        upload_a = st.file_uploader("Upload Frame A", type=UPLOAD_TYPES, key=f"frame_a_{upload_generation}", help=f"Supported: {', '.join(UPLOAD_TYPES).upper()}")
        image_a = upload_image(upload_a)
    with b_col:
        st.markdown("### END FRAME")
        upload_b = st.file_uploader("Upload Frame B", type=UPLOAD_TYPES, key=f"frame_b_{upload_generation}", help=f"Supported: {', '.join(UPLOAD_TYPES).upper()}")
        image_b = upload_image(upload_b)
    if image_a is not None:
        a_col.image(image_a, caption=f"Frame A · {image_a.width} × {image_a.height}", **IMAGE_STRETCH)
    if image_b is not None:
        b_col.image(image_b, caption=f"Frame B · {image_b.width} × {image_b.height}", **IMAGE_STRETCH)
    if image_a is None or image_b is None:
        st.caption("Upload both frames to enable inbetween generation.")
    else:
        st.caption("Different input dimensions and aspect ratios are fitted into the model canvas while preserving shape.")

    current_signature = input_signature(upload_a, upload_b)
    if st.button("Generate Inbetween Frame", type="primary", use_container_width=True, key=f"generate_{upload_generation}"):
        # Never leave a previous image/statistics visible after a new attempt.
        st.session_state.pop("result", None)
        if upload_a is None:
            st.error("Upload Frame A before generating.")
        elif upload_b is None:
            st.error("Upload Frame B before generating.")
        elif image_a is None or image_b is None:
            st.error("One or both uploaded images could not be decoded. Replace the invalid file and try again.")
        elif model is None:
            st.error(model_error or "The trained model is unavailable.")
        else:
            try:
                with st.spinner("Generating inbetween frame..."):
                    inference = validate_inference_result(
                        predict_image_pair(model, image_a, image_b, DEVICE)
                    )
                    inference_seconds = inference["inference_seconds"]
                    generated_array = inference["generated_frame"]
                    generated_image = Image.fromarray(generated_array)
                    comparison_image = make_comparison(image_a, generated_image, image_b)

                    generated_buffer = io.BytesIO()
                    generated_image.save(generated_buffer, format="PNG")
                    comparison_buffer = io.BytesIO()
                    comparison_image.save(comparison_buffer, format="PNG")
                    generated_bytes = generated_buffer.getvalue()
                    comparison_bytes = comparison_buffer.getvalue()
                    if not generated_bytes or not comparison_bytes:
                        raise OSError("PNG encoding returned empty image data.")

                    generated_dir = PROJECT_ROOT / "outputs" / "generated"
                    comparison_dir = PROJECT_ROOT / "outputs" / "comparisons"
                    generated_dir.mkdir(parents=True, exist_ok=True)
                    comparison_dir.mkdir(parents=True, exist_ok=True)
                    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
                    generated_path = generated_dir / f"inbetween_{run_id}.png"
                    comparison_path = comparison_dir / f"inbetween_{run_id}_comparison.png"
                    generated_path.write_bytes(generated_bytes)
                    comparison_path.write_bytes(comparison_bytes)

                st.session_state.result = {
                    "schema_version": 2,
                    "image": generated_array,
                    "png_bytes": generated_bytes,
                    "comparison_bytes": comparison_bytes,
                    "signature": current_signature,
                    "time": inference_seconds,
                    "prediction_stats": inference["prediction_stats"],
                    "image_stats": inference["image_stats"],
                    "start_size": image_a.size,
                    "end_size": image_b.size,
                    "model": getattr(model, "architecture", model.__class__.__name__),
                    "generated_path": str(generated_path),
                    "comparison_path": str(comparison_path),
                }
                st.success("Generation completed successfully.")
            except Exception as exc:
                st.error(f"Generation failed: {type(exc).__name__}: {exc}")

    result = st.session_state.get("result")
    # Streamlit can retain session data across code hot reloads. Discard any
    # result made under an earlier schema instead of indexing missing fields.
    required_result_fields = {
        "image", "png_bytes", "comparison_bytes", "signature", "time",
        "prediction_stats", "image_stats", "model", "generated_path",
    }
    if result is not None and (
        not isinstance(result, dict)
        or result.get("schema_version") != 2
        or not required_result_fields.issubset(result)
    ):
        st.session_state.pop("result", None)
        result = None
    if result and result.get("signature") == current_signature and image_a is not None and image_b is not None:
        generated_display = Image.fromarray(result["image"])
        st.divider()
        st.subheader("Generated Result")
        st.image(generated_display, caption="Generated Inbetween Frame", **IMAGE_STRETCH)
        st.subheader("Comparison")
        st.image(result["comparison_bytes"], caption="Frame A | AI Generated Frame | Frame B", **IMAGE_STRETCH)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Input resolution", f"{image_a.width}×{image_a.height} / {image_b.width}×{image_b.height}")
        m2.metric("Output resolution", f"{result['image'].shape[1]}×{result['image'].shape[0]}")
        m3.metric("Inference time", f"{result['time']:.3f} s")
        m4.metric("Model", result["model"])
        d1, d2 = st.columns(2)
        d1.download_button("Download Generated Frame", data=result["png_bytes"], file_name="generated_inbetween_frame.png", mime="image/png", key=f"download_generated_{upload_generation}", use_container_width=True)
        d2.download_button("Download Comparison", data=result["comparison_bytes"], file_name="inbetween_comparison.png", mime="image/png", key=f"download_comparison_{upload_generation}", use_container_width=True)
        st.caption(f"Saved generated PNG: {result['generated_path']}")
        st.caption("The result is a model prediction. Uploaded images have no ground-truth frame, so evaluation metrics are not calculated.")
        st.caption(result.get("geometry_mode", "Raster-guided inference (estimated geometry)"))
        with st.expander("Output diagnostics"):
            prediction_stats = result["prediction_stats"]
            image_stats = result["image_stats"]
            st.caption(
                f"Raw sigmoid prediction: shape {prediction_stats['shape']}, "
                f"{prediction_stats['dtype']}, min {prediction_stats['min']:.4f}, "
                f"max {prediction_stats['max']:.4f}, mean {prediction_stats['mean']:.4f}, "
                f"std {prediction_stats['std']:.4f}. PNG: mode {image_stats['mode']}, "
                f"size {image_stats['size'][0]}×{image_stats['size'][1]}, "
                f"pixels {image_stats['pixel_min']}–{image_stats['pixel_max']}."
            )
            bins = prediction_stats.get("value_bins_pct", {})
            if bins:
                st.caption("Raw prediction pixel distribution: " + " · ".join(f"{band}: {percent:.1f}%" for band, percent in bins.items()))


def show_geometry():
    heading("INSPECT", "Geometry Analyzer", "Visualize image-derived geometry. Upload geometry remains estimated inference-time geometry.")
    st.info("Estimated inference-time geometry · This is derived from image edges and is not ML240 ground-truth annotation.")
    upload = st.file_uploader("Upload a cartoon line-art frame", type=["png", "jpg", "jpeg", "bmp"], key="geometry_upload")
    image = upload_image(upload)
    if image is None: return
    try:
        line, vertex, graph, canvas, _ = image_to_inputs(image)
        rgb = np.asarray(image)
        edges = extract_edges(canvas)
        vertex_map = vertex.squeeze().numpy()
        graph_map = graph.squeeze().numpy()
        overlay = canvas.copy()
        overlay[graph_map > .2] = (70, 145, 255)
        overlay[vertex_map > .2] = (255, 105, 82)
        cols = st.columns(3)
        cols[0].image(rgb, caption="Original frame", **IMAGE_STRETCH)
        cols[1].image(edges, caption="Estimated edge map", clamp=True, **IMAGE_STRETCH)
        cols[2].image(vertex_map, caption="Estimated vertex heatmap", clamp=True, **IMAGE_STRETCH)
        cols2 = st.columns(2)
        cols2[0].image(graph_map, caption="Estimated graph-edge map", clamp=True, **IMAGE_STRETCH)
        cols2[1].image(overlay, caption="Geometry overlay · orange vertices / blue graph edges", **IMAGE_STRETCH)
    except Exception as exc: st.error(f"Geometry analysis failed: {exc}")


def sequence_counts():
    counts = {}
    for split in ("train", "val", "test"):
        folder = PROJECT_ROOT / "dataset" / split
        counts[split] = len([p for p in folder.iterdir() if p.is_dir()]) if folder.is_dir() else None
    return counts


def show_model(model):
    heading("MODEL", "Model & Performance", "Verified architecture details and held-out sample evaluation.")
    steps = ["Endpoint line maps", "ML240 / estimated geometry maps", "Separate residual encoders", "Gated multi-scale fusion", "Skip-connected decoder", "Middle line-art head"]
    cols = st.columns(3)
    for i, label in enumerate(steps):
        with cols[i % 3]: st.markdown(f'<div class="step"><b>STAGE {i+1}</b>{label}</div>', unsafe_allow_html=True)
    st.write("")
    if model:
        params = sum(p.numel() for p in model.parameters())
        info = [("Architecture", model.architecture), ("Input channels", str(model.input_channels)), ("Output channels", str(model.output_channels)), ("Parameters", f"{params:,}"), ("Image size", f"{IMAGE_SIZE} × {IMAGE_SIZE}"), ("Device", str(DEVICE))]
    else: info = [("Architecture", "Unavailable until model loads"), ("Parameters", "Not evaluated"), ("Device", str(DEVICE))]
    cols = st.columns(3)
    for col, (label, value) in zip(cols, info): col.metric(label, value)
    st.markdown("#### Model contract and training")
    st.write("The trained network receives seven 256 × 256 channels: Frame A line, A vertex map, A graph-edge map, Frame B line, B vertex map, B graph-edge map, and midpoint time. It predicts one grayscale middle-frame line-intensity map. Training uses adjacent ML240 frames and their geometry JSON labels.")
    st.write("The configured objective combines L1 reconstruction, weighted MSE, finite-difference edge loss, and a local SSIM-style structural loss. The training script uses AdamW and validation-driven learning-rate reduction.")
    st.caption("Configured loss weights: " + " · ".join(f"{name.upper()} {weight:g}" for name, weight in LOSS_WEIGHTS.items()))
    counts = sequence_counts()
    st.markdown("#### Prepared dataset sequences")
    cols = st.columns(3)
    for col, split in zip(cols, ("train", "val", "test")):
        col.metric(split.upper(), "Not available" if counts[split] is None else counts[split])
    st.caption("Sequence-level splits are kept separate to avoid frame-level leakage. Training geometry is read from ML240 label JSON; inference geometry is estimated from the uploaded raster.")


def show_evaluation():
    st.markdown("#### Held-out test sample")
    st.caption("These values come from the latest actual CLI test evaluation. They describe that selected dataset sample and are not guarantees for uploaded images.")
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    report_path = metrics_dir / "latest_metrics.json"
    comparison_dir = PROJECT_ROOT / "outputs" / "comparisons"
    # Only show evaluation comparisons here; studio upload comparisons have no
    # ground-truth middle frame and must not be labeled as held-out results.
    comparison = max(comparison_dir.glob("test_*_comparison.png"), key=lambda p: p.stat().st_mtime, default=None) if comparison_dir.is_dir() else None
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            cols = st.columns(5)
            for col, key, label, fmt in zip(cols, ("MAE", "MSE", "PSNR_dB", "SSIM", "ChamferDistance"), ("MAE", "MSE", "PSNR", "SSIM", "Chamfer distance"), (".5f", ".6f", ".2f", ".4f", ".3f")):
                value = report.get(key)
                col.metric(label, "Not evaluated" if value is None else format(value, fmt))
        except Exception as exc: st.warning(f"Could not read the latest metrics file: {exc}")
    else: st.info("Not evaluated · run `python test.py --index 0` to populate held-out metrics.")
    if comparison:
        st.image(str(comparison), caption=f"Latest held-out comparison · {comparison.name}", **IMAGE_STRETCH)
    else: st.caption("No test comparison has been generated yet.")
    st.caption("Metrics depend on the held-out subset, image resolution, checkpoint, and threshold used for the binary-line Chamfer distance.")


def show_how_it_works():
    heading("PIPELINE", "How It Works", "The model predicts a plausible missing drawing between two cartoon keyframes.")
    stages = [
        ("01 · INPUT", "Frame A + Frame B", "Upload two readable cartoon line-art frames. Different image dimensions are fitted to the model canvas while preserving aspect ratio."),
        ("02 · PREPROCESS", "Normalize and map", "Each frame becomes a grayscale line-intensity map. Inference-time vertex and graph-edge features are estimated from its contours."),
        ("03 · NETWORK", "Two feature branches", "A raster encoder and a geometry encoder extract separate multi-scale features, then gated fusion combines them."),
        ("04 · PREDICT", "Middle-frame output", "The decoder predicts a one-channel 256 × 256 line-intensity image for the midpoint."),
        ("05 · POSTPROCESS", "Display and download", "The output is converted to grayscale PNG, restored to Frame A’s dimensions, and saved with a three-frame comparison."),
    ]
    for kicker, title, description in stages:
        st.markdown(f'<div class="card" style="margin:.65rem 0"><div class="eyebrow">{kicker}</div><h3>{title}</h3><p>{description}</p></div>', unsafe_allow_html=True)
    st.info("Training uses ML240 ground-truth geometry labels. Uploaded-image inference estimates geometry from the image; it is not ground-truth annotation.")


def show_about():
    heading("PROJECT", "About DeepLine AI", "Final-year academic project · Cartoon line inbetweening")
    sections = [
        ("Project", "Deep Geometrized Cartoon Line Inbetweening"),
        ("Problem", "Given two endpoint cartoon drawings, synthesize a plausible missing intermediate line-art frame."),
        ("Objective", "Use endpoint raster evidence together with geometry-aware features to reconstruct a middle frame."),
        ("Dataset", "A reduced, sequence-level subset of ML240 / MixamoLine240 with geometry label JSON files."),
        ("Method", "Training raster pairs use actual ML240 vertex and graph labels. Upload inference estimates geometry from image contours."),
        ("Architecture", "Compact residual dual-branch encoder-decoder with channel attention, gated fusion, and multi-scale skip features."),
        ("Evaluation", "MAE, MSE, PSNR, SSIM and binary-line Chamfer distance against held-out ground-truth middle frames."),
        ("Technologies", "Python, PyTorch, OpenCV, NumPy, Pillow and Streamlit."),
        ("Limitations", "A small reduced dataset and 256-pixel training canvas limit detail and motion generalization. Inference geometry is approximate."),
        ("Future work", "Expand sequence coverage, add explicit temporal correspondence supervision, and evaluate across multiple seeds and animation styles."),
        ("License", "Check ML240 / MixamoLine240 terms before redistribution; cite the AnimeInbet paper and dataset source in academic work."),
    ]
    for title, body in sections:
        with st.expander(title, expanded=title in ("Problem", "Method")): st.write(body)
    st.info("This implementation is a practical ML240-based geometry-aware system inspired by AnimeInbet and is not an exact reproduction of the original ICCV 2023 architecture.")


st.sidebar.markdown("# ✦ DeepLine AI")
st.sidebar.caption("GEOMETRY-AWARE LINE INBETWEENING")
pages = ["HOME", "INBETWEENING STUDIO", "HOW IT WORKS", "GEOMETRY ANALYZER", "MODEL & PERFORMANCE", "ABOUT"]
page = st.sidebar.radio("NAVIGATION", pages, key="page")
st.sidebar.divider()
st.sidebar.caption(f"Compute · {DEVICE}")
checkpoint_signature = "missing"
if MODEL_PATH.is_file():
    checkpoint_stat = MODEL_PATH.stat()
    checkpoint_signature = f"{checkpoint_stat.st_size}:{checkpoint_stat.st_mtime_ns}"
model, model_error = load_model(str(MODEL_PATH), str(DEVICE), checkpoint_signature)
if model:
    st.sidebar.success("Model ready")
elif page != "HOME": st.sidebar.warning("Model unavailable")

if page == "HOME": show_home()
elif page == "INBETWEENING STUDIO": show_studio(model, model_error)
elif page == "HOW IT WORKS": show_how_it_works()
elif page == "GEOMETRY ANALYZER": show_geometry()
elif page == "MODEL & PERFORMANCE":
    show_model(model)
    show_evaluation()
else: show_about()

