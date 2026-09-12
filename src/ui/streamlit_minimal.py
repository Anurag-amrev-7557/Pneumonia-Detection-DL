"""
PULMO·AI — Clinical Chest X-Ray Pneumonia Detection
Minimal, professional diagnostic interface powered by dual-backbone ensemble.
"""

import sys
import json
from pathlib import Path
from typing import Any
import base64
from io import BytesIO

import numpy as np
import cv2
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# Setup
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config.settings import settings

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION & MINIMAL STYLING
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PULMO·AI | CXR Diagnostics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, minimal CSS
st.markdown("""
<style>
    :root {
        --primary: #2563eb;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg: #ffffff;
        --surface: #f8f9fa;
        --border: #e5e7eb;
        --text: #111827;
        --text-light: #6b7280;
    }

    body, html {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    .stApp {
        background-color: var(--bg) !important;
        max-width: 1200px;
        margin: 0 auto;
    }

    h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; margin: 0 0 0.5rem 0; }
    h2 { font-size: 1.5rem; font-weight: 600; margin: 1.5rem 0 1rem 0; }
    h3 { font-size: 1.1rem; font-weight: 600; margin: 1rem 0 0.5rem 0; }
    p { font-size: 0.95rem; line-height: 1.6; color: var(--text-light); }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-light);
        font-weight: 600;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0.5rem 0;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }

    .badge-success {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success);
        border: 1px solid var(--success);
    }

    .badge-warning {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning);
        border: 1px solid var(--warning);
    }

    .badge-danger {
        background: rgba(239, 68, 68, 0.1);
        color: var(--danger);
        border: 1px solid var(--danger);
    }

    .finding-card {
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }

    .finding-normal {
        background: rgba(16, 185, 129, 0.05);
        border: 2px solid var(--success);
    }

    .finding-pneumonia {
        background: rgba(239, 68, 68, 0.05);
        border: 2px solid var(--danger);
    }

    .finding-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .finding-subtitle {
        font-size: 0.95rem;
        color: var(--text-light);
        margin: 0;
    }

    /* Streamlit overrides */
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
    }

    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load TFLite model once and cache it."""
    try:
        from src.utils.model_loader import ensure_models_downloaded, MODELS_DIR
        from src.models.tflite_inference import TFLiteDetector

        # Ensure models are downloaded
        ok = ensure_models_downloaded()
        if not ok:
            raise RuntimeError("Failed to download models")

        model_path = MODELS_DIR / "best_model.tflite"
        secondary_path = MODELS_DIR / "densenet121_best.tflite"

        if not model_path.exists():
            raise RuntimeError(f"Model file not found: {model_path}")

        detector = TFLiteDetector(
            model_path=model_path,
            secondary_model_path=secondary_path if secondary_path.exists() else None
        )
        return detector, None
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def blend_heatmap(original: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    """Blend heatmap over original image."""
    h, w = original.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_scaled = (np.clip(heatmap_resized, 0, 1) * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    blended = cv2.addWeighted(
        original, 1.0 - alpha,
        heatmap_colored, alpha,
        0
    )
    return blended

def get_demo_image(label: str) -> np.ndarray | None:
    """Load a demo image from test set."""
    try:
        if label == "normal":
            test_dir = Path("data/test/NORMAL")
            if test_dir.exists():
                images = list(test_dir.glob("*.jpeg"))
                if images:
                    return np.array(Image.open(images[0]))
        else:
            test_dir = Path("data/test/PNEUMONIA")
            if test_dir.exists():
                images = list(test_dir.glob("*.jpeg"))
                if images:
                    return np.array(Image.open(images[0]))
    except Exception as e:
        st.warning(f"Could not load demo: {e}")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("# 🫁 PULMO·AI")
    st.markdown("**Clinical Pneumonia Detection** | Powered by dual-backbone deep learning ensemble")
    st.divider()

    # Load model
    detector, error = load_model()
    if error:
        st.error(f"❌ Model Loading Failed\n\n{error}")
        st.stop()

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        confidence_threshold = st.slider(
            "Decision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probability threshold for pneumonia classification"
        )
        show_heatmap = st.checkbox("Show Grad-CAM Heatmap", value=True)
        gradcam_opacity = st.slider("Heatmap Opacity", 0.0, 1.0, 0.4) if show_heatmap else 0.4

    # Main content area
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.subheader("📤 Input")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Upload Image", "Use Demo Sample"],
            label_visibility="collapsed"
        )

        image = None
        if input_method == "Upload Image":
            uploaded = st.file_uploader(
                "Upload a chest X-ray (JPEG, PNG)",
                type=["jpeg", "jpg", "png"]
            )
            if uploaded:
                image = np.array(Image.open(uploaded).convert("RGB"))
        else:
            demo_col1, demo_col2 = st.columns(2)
            with demo_col1:
                if st.button("📋 Normal Sample"):
                    image = get_demo_image("normal")
                    if image is None:
                        st.warning("No demo images available")
            with demo_col2:
                if st.button("🫁 Pneumonia Sample"):
                    image = get_demo_image("pneumonia")
                    if image is None:
                        st.warning("No demo images available")

        # Display input image
        if image is not None:
            st.image(image, use_column_width=True)

    with col2:
        st.subheader("📊 Results")

        if image is not None:
            # Run inference
            with st.spinner("🔍 Analyzing..."):
                try:
                    result = detector.predict(
                        image,
                        confidence_threshold=confidence_threshold,
                        return_gradcam=show_heatmap
                    )

                    # Determine finding
                    is_pneumonia = result.get("is_pneumonia", False)
                    confidence = result.get("confidence", 0.0)
                    processing_time = result.get("processing_time", 0.0)

                    # Finding badge
                    if is_pneumonia:
                        color = "danger" if confidence >= 0.65 else "warning"
                        finding = "🫁 PNEUMONIA DETECTED"
                        finding_class = "finding-pneumonia"
                    else:
                        color = "success"
                        finding = "✓ NORMAL / CLEAR"
                        finding_class = "finding-normal"

                    st.markdown(f"""
                    <div class="{finding_class}">
                        <div class="finding-title">{finding}</div>
                        <div class="finding-subtitle">Confidence: {confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Per-backbone breakdown
                    st.markdown("##### Consensus")
                    breakdown = result.get("breakdown", {})
                    if breakdown:
                        cols = st.columns(len(breakdown))
                        for idx, (model_name, probs) in enumerate(breakdown.items()):
                            with cols[idx]:
                                pneumonia_prob = probs.get("Pneumonia", 0.0)
                                st.metric(
                                    model_name.split()[0],
                                    f"{pneumonia_prob:.1%}",
                                    delta=f"{(pneumonia_prob - confidence) * 100:.1f}pp"
                                )
                    
                    # Metadata
                    st.markdown("##### Details")
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Processing", f"{processing_time:.0f}ms")
                    with cols[1]:
                        st.metric("Threshold", f"{confidence_threshold:.0%}")
                    with cols[2]:
                        is_borderline = result.get("is_borderline", False)
                        status = "⚠️ Equivocal" if is_borderline else "✓ Clear"
                        st.metric("Status", status)

                    # Grad-CAM display
                    if show_heatmap and result.get("raw_heatmap") is not None:
                        st.markdown("---")
                        st.markdown("##### Attention Map")
                        heatmap_blended = blend_heatmap(
                            image,
                            result["raw_heatmap"],
                            gradcam_opacity
                        )
                        st.image(heatmap_blended, use_column_width=True)

                    # Export button
                    report = {
                        "timestamp": str(datetime.now()),
                        "finding": finding,
                        "confidence": float(confidence),
                        "is_borderline": bool(result.get("is_borderline", False)),
                        "breakdown": breakdown,
                        "processing_time_ms": float(processing_time)
                    }
                    
                    st.download_button(
                        label="📥 Download Report (JSON)",
                        data=json.dumps(report, indent=2),
                        file_name=f"pulmo_report_{datetime.now():%Y%m%d_%H%M%S}.json",
                        mime="application/json"
                    )

                except Exception as e:
                    st.error(f"Inference failed: {e}")
        else:
            st.info("👆 Upload or select a sample to begin")

    # Footer
    st.divider()
    st.markdown("""
    ---
    **PULMO·AI** v2.4 | Clinical decision support tool
    
    ⚠️ **Clinical Disclaimer**: Results are for research and diagnostic assistance only.
    Always consult a board-certified radiologist for clinical interpretation.
    """)

if __name__ == "__main__":
    from datetime import datetime
    main()
