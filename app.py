# ============================================================
# MCC Diagnostic App – Academic Research Prototype
# ============================================================

# -------------------------------
# ENVIRONMENT SAFETY (MUST BE FIRST)
# -------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -------------------------------
# IMPORTS
# -------------------------------
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
from fpdf import FPDF
import tempfile

# -------------------------------
# CONFIGURATION
# -------------------------------
IMG_SIZE = 224
THRESHOLD = 0.35  # MCC-safe ROC threshold

# Sigmoid output = P(Non-MCC)
IDX_TO_CLASS = {0: "MCC", 1: "Non-MCC"}

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(
    page_title="MCC Diagnostic App",
    page_icon="🩺",
    layout="centered"
)

# -------------------------------
# HEADER
# -------------------------------
st.title("🩺 MCC Diagnostic App")
st.subheader("AI-Assisted Merkel Cell Carcinoma Risk Assessment")

st.markdown("""
⚠️ **Academic Research Prototype – NOT a Medical Device**

This system is developed strictly for **research and educational purposes**.  
It is **NOT approved for clinical diagnosis**.  
Final decisions must always be made by certified dermatologists.
""")

st.divider()

# -------------------------------
# ABOUT MCC & MODEL
# -------------------------------
with st.expander("📘 About MCC & the AI Model", expanded=False):
    st.markdown("""
**Merkel Cell Carcinoma (MCC)** is a rare but highly aggressive neuroendocrine skin cancer.  
Early detection is critical due to its high metastatic potential.

### 🔬 Model Overview
- Backbone: **Vision Transformer / CNN Hybrid**
- Input: Dermoscopic skin lesion images
- Output: **Binary classification**
  - MCC
  - Non-MCC
- Threshold selected via **ROC curve optimization**
- Deployment: **Inference-only TensorFlow SavedModel**
""")

st.divider()

# -------------------------------
# LOAD MODEL (KERAS 3 SAFE)
# -------------------------------
@st.cache_resource
def load_model():
    model_dir = "mcc_model_savedmodel"
    if not os.path.exists(model_dir):
        st.error("❌ Model folder not found.")
        return None

    return tf.keras.layers.TFSMLayer(
        model_dir,
        call_endpoint="serving_default"
    )

model = load_model()
if model is None:
    st.stop()

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
st.header("📤 Upload Skin Lesion Image")

uploaded_file = st.file_uploader(
    "Supported formats: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    # -------------------------------
    # PREPROCESSING
    # -------------------------------
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # -------------------------------
    # BASIC VISUAL FEATURE EXTRACTION
    # -------------------------------
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lesion_area = perimeter = circularity = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        lesion_area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            circularity = (4 * np.pi * lesion_area) / (perimeter ** 2)

    st.divider()

    # -------------------------------
    # FEATURE DISPLAY
    # -------------------------------
    st.subheader("🧬 Extracted Visual Indicators (Approximate)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Lesion Area", f"{lesion_area:.0f} px²")
    c2.metric("Perimeter", f"{perimeter:.2f} px")
    c3.metric("Circularity", f"{circularity:.3f}")
    st.caption("⚠️ Visual explanation only – not diagnostic features.")

    st.divider()

    # -------------------------------
    # MODEL INFERENCE
    # -------------------------------
    with st.spinner("Analyzing lesion using AI model..."):
        output = model(img_array)
        prob_non_mcc = float(list(output.values())[0][0][0])

    prob_mcc = 1.0 - prob_non_mcc

    # -------------------------------
    # RISK INTERPRETATION
    # -------------------------------
    if prob_mcc >= 0.65:
        risk, badge = "HIGH", "🚨"
    elif prob_mcc >= THRESHOLD:
        risk, badge = "MEDIUM", "⚠️"
    else:
        risk, badge = "LOW", "✅"

    st.header("📊 Prediction Results")
    r1, r2 = st.columns(2)
    r1.metric("MCC Probability", f"{prob_mcc * 100:.2f}%")
    r2.metric("Benign Probability", f"{prob_non_mcc * 100:.2f}%")
    st.markdown(f"### {badge} **Risk Level: {risk}**")

    if risk == "HIGH":
        st.error("Immediate dermatological evaluation is strongly advised.")
    elif risk == "MEDIUM":
        st.warning("Clinical review is recommended.")
    else:
        st.success("Routine monitoring is suggested.")

    st.divider()

    # -------------------------------
    # PDF REPORT
    # -------------------------------
    st.header("📄 Download Diagnostic Report")

    def safe(text: str) -> str:
        return (
            text.replace("–", "-")
                .replace("—", "-")
                .replace("’", "'")
                .replace("“", '"')
                .replace("”", '"')
        )

    def generate_pdf():
        pdf = FPDF()
        pdf.set_auto_page_break(True, 15)
        pdf.add_page()

        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, safe("MCC Diagnostic Report"), ln=True, align="C")

        pdf.set_font("Arial", "", 10)
        pdf.cell(
            0, 8,
            safe(f"Date: {datetime.now().strftime('%d %b %Y | %H:%M')}"),
            ln=True
        )

        pdf.ln(4)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, safe("Merkel Cell Carcinoma (MCC)"), ln=True)

        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(
            0, 6,
            safe(
                "Merkel Cell Carcinoma (MCC) is a rare and aggressive skin cancer "
                "originating from neuroendocrine cells. Early detection is critical."
            )
        )

        pdf.ln(4)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            pdf.image(tmp.name, x=30, w=150)

        pdf.ln(6)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, safe("Prediction Summary"), ln=True)

        pdf.set_font("Arial", "", 10)
        pdf.cell(90, 8, safe("MCC Probability:"), 0, 0)
        pdf.cell(0, 8, f"{prob_mcc * 100:.2f} %", ln=True)
        pdf.cell(90, 8, safe("Benign Probability:"), 0, 0)
        pdf.cell(0, 8, f"{prob_non_mcc * 100:.2f} %", ln=True)
        pdf.cell(90, 8, safe("Risk Level:"), 0, 0)
        pdf.cell(0, 8, safe(risk), ln=True)

        pdf.ln(4)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, safe("Visual Indicators"), ln=True)

        pdf.set_font("Arial", "", 10)
        pdf.cell(90, 8, safe("Lesion Area:"), 0, 0)
        pdf.cell(0, 8, f"{lesion_area:.0f}", ln=True)
        pdf.cell(90, 8, safe("Perimeter:"), 0, 0)
        pdf.cell(0, 8, f"{perimeter:.2f}", ln=True)
        pdf.cell(90, 8, safe("Circularity:"), 0, 0)
        pdf.cell(0, 8, f"{circularity:.3f}", ln=True)

        pdf.ln(6)
        pdf.set_font("Arial", "I", 8)
        pdf.multi_cell(
            0, 5,
            safe(
                "This report is generated by an AI research system.\n"
                "NOT a medical diagnosis."
            )
        )

        return pdf.output(dest="S").encode("latin-1")

    st.download_button(
        "📥 Download Diagnostic Report (PDF)",
        generate_pdf(),
        "MCC_Diagnostic_Report.pdf",
        "application/pdf"
    )

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("© Engineering Research Prototype · AI-Assisted MCC Risk Assessment")
