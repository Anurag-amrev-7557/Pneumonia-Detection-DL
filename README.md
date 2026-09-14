# 🫁 PULMO·AI™ — Radiologist-Level Pneumonia Detection Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![TensorFlow / Keras 3](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-3.15-FF6F00.svg?logo=tensorflow&logoColor=white)](https://keras.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests Passing](https://img.shields.io/badge/tests-28%2F28%20passing-10B981.svg?logo=pytest&logoColor=white)](tests/)
[![Normal Specificity](https://img.shields.io/badge/Normal%20Specificity-98.97%25-10B981.svg)](#-clinical-benchmark--performance)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9935-6366F1.svg)](#-clinical-benchmark--performance)
[![License: MIT](https://img.shields.io/badge/License-MIT-38BDF8.svg)](LICENSE)

> **PULMO·AI™** is a clinical-grade deep learning diagnostic platform for automated pneumonia triage from frontal digital chest radiographs (CXR). Powered by a **CheXNet-inspired Dual-Backbone Ensemble (ResNet-50 + DenseNet-121)** with Grad-CAM explainability and real-time PACS inspection tools, PULMO·AI delivers high-sensitivity screening ($92.4\%$) alongside exceptional specificity on normal radiographs ($98.97\%$), minimizing false alarms.

---

## 📑 Table of Contents
1. [Key Capabilities](#-key-capabilities)
2. [Dual-Backbone CheXNet Architecture](#-dual-backbone-chexnet-architecture)
3. [Clinical Benchmark & Performance](#-clinical-benchmark--performance)
4. [Explainable AI (Grad-CAM Saliency)](#-explainable-ai-grad-cam-saliency)
5. [Clinical PACS Inspection Controls](#-clinical-pacs-inspection-controls)
6. [Repository Structure](#-repository-structure)
7. [Quick Start Guide](#-quick-start-guide)
8. [Unified CLI Reference (`main.py`)](#-unified-cli-reference-mainpy)
9. [REST API Service (FastAPI)](#-rest-api-service-fastapi)
10. [Automated Test Suite](#-automated-test-suite)
11. [License & Medical Disclaimer](#-license--medical-disclaimer)

---

## 🌟 Key Capabilities

* **CheXNet Dual-Backbone Ensemble**: Fuses deep residual representations (ResNet-50) with dense multi-scale feature reuse (DenseNet-121) to capture both coarse consolidations and delicate interstitial opacities.
* **Exceptional Normal Specificity (98.97%)**: Prevents false-positive alarms on normal adult lung markings and anatomical variations.
* **Interpretable Radiologic AI (Grad-CAM)**: Generates high-resolution saliency maps localized to lung fields, verifying that model predictions are driven by actual pulmonary infiltrates rather than scanner borders or technician lead markers.
* **PACS Workstation Interface**: Real-time Streamlit diagnostic viewer featuring:
  * ✂️ **Auto-Crop Margins (6–7%)**: Automatically strips peripheral technician letter markers ("R", "L") and scanner edge noise.
  * 🔄 **Contrast Inversion**: Normalizes photonegative web images to radiological DICOM standard (MONOCHROME2).
  * 🎚️ **Adjustable Sensitivity Threshold**: Allows clinicians to toggle between screening mode ($0.35$ high sensitivity) and diagnostic confirmation ($0.65$ high specificity).
  * 🧪 **1-Click Demo Scans**: Instant evaluation of curated Normal, Bacterial Pneumonia, Viral Pneumonia, and subtle infiltrate radiographs.
* **Dual Runtime Engines**:
  * **Full TensorFlow Engine**: For local workstation deployments with GPU acceleration and full Grad-CAM backpropagation.
  * **Quantized LiteRT / TFLite Engine**: For instant, low-latency, low-RAM deployments (e.g. Streamlit Cloud).

---

## 🧠 Dual-Backbone CheXNet Architecture

Inspired by Stanford's landmark CheXNet paper (*Rajpurkar et al., 2017*), PULMO·AI combines two structurally divergent architectures via equal-weight soft-voting consensus:

```text
                     Frontal Chest Radiograph (224x224x3)
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
     ┌──────────────┐                                    ┌──────────────┐
     │  ResNet-50   │                                    │ DenseNet-121 │
     │  (Residual)  │                                    │  (CheXNet)   │
     └──────┬───────┘                                    └──────┬───────┘
            │                                                   │
     P(Pneumonia)_res                                    P(Pneumonia)_dense
            │                                                   │
            └─────────────────────────┬─────────────────────────┘
                                      ▼
                      Soft-Voting Consensus Ensemble
               P_final = 0.5 * P_res + 0.5 * P_dense
                                      │
                                      ▼
                        Diagnostic Classification:
                           [ NORMAL vs PNEUMONIA ]
```

1. **DenseNet-121 (CheXNet Backbone)**: Every layer receives direct inputs from all preceding layers ($x_l = H_l([x_0, x_1, \dots, x_{l-1}])$). This architecture preserves fine-grained bronchial structures and suppresses background noise on healthy lung parenchyma.
2. **ResNet-50**: Deep residual skip connections ($\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$) allow robust feature extraction across dense lobar consolidations.

---

## 📊 Clinical Benchmark & Performance

Evaluated on the certified held-out test cohort:

| Metric | Standalone ResNet-50 | Standalone DenseNet-121 | **Dual CheXNet Ensemble** | Clinical Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Balanced Accuracy** | 93.46% | 96.16% | **95.68%** | Robust cross-class balance |
| **ROC-AUC** | 0.9879 | 0.9972 | **0.9935** | Excellent discriminative power |
| **Macro F1-Score** | 0.9074 | 0.9442 | **0.9302** | Balanced precision & recall |
| **Specificity (Normal)** | 96.45% | 97.97% | **98.97%** | **Near-zero false alarms on normal lungs** |
| **Sensitivity (Pneumonia)** | 90.47% | 94.36% | **92.38%** | Reliable detection of acute opacity |

---

## 🔍 Explainable AI (Grad-CAM Saliency)

To ensure clinical trustworthiness, PULMO·AI computes gradient-weighted class activation maps at the final convolutional stage:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right), \quad \text{where } \alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial Y^c}{\partial A_{i,j}^k}$$

This transparently highlights whether model activations correspond to:
* **Focal Alveolar Consolidation** (typical of lobar bacterial pneumonia)
* **Diffuse / Bilateral Interstitial Infiltrates** (typical of viral pneumonia)
* Or whether an alert was triggered by an extraneous peripheral artifact.

---

## 🛠️ Clinical PACS Inspection Controls

Radiographs sourced from external facilities, scanners, or web resources often suffer from photometric variations:
* **Auto-Crop Margins**: Strips up to 7% of peripheral border artifacts to isolate lung fields from technician letter markers ("L", "R").
* **Contrast Inversion**: Reverses inverted (photonegative) scans back to standard DICOM MONOCHROME2.
* **Sensitivity Calibration**: Move from screening mode (35% threshold) to confirmatory triage (65% threshold).

---

## 📁 Repository Structure

```
├── data/
│   └── test/                      # Curated test & demo radiographs
├── models/
│   ├── current/                   # Active production models
│   │   ├── best_model.h5          # ResNet-50 primary backbone (211 MB)
│   │   ├── densenet121_best.h5    # DenseNet-121 CheXNet backbone (36 MB)
│   │   ├── ensemble_metadata.json # Certified consensus weights & test metrics
│   │   ├── model_metadata.json    # Training hyperparameters
│   │   └── training_history.json  # Loss & accuracy curves
│   └── README.md                  # Detailed model documentation
├── scripts/
│   ├── download_weights.py        # Automated weight downloader with SHA-256
│   ├── evaluate_model.py          # Benchmark evaluation script
│   ├── diagnose_overfitting.py    # Generalization audit script
│   └── verify_zero_leakage.py     # Patient isolation audit tool
├── src/
│   ├── api/
│   │   └── server.py              # Production FastAPI REST backend
│   ├── models/
│   │   ├── architectures.py       # ResNet, DenseNet, VGG, CNN definitions
│   │   ├── grad_cam.py            # High-resolution Grad-CAM visualizer
│   │   ├── inference.py           # Full TensorFlow dual-backbone detector
│   │   ├── tflite_inference.py    # Lightweight LiteRT runtime
│   │   └── training.py            # Model training & optimization loop
│   ├── ui/
│   │   └── streamlit_app.py       # Clinical PACS Workstation UI
│   └── utils/
│       ├── image_processing.py    # CLAHE, normalization, transforms
│       ├── model_loader.py        # Cloud weights & sample manager
│       └── radiology.py           # Radiologic heuristics & colormaps
├── tests/                         # Comprehensive pytest suite (28 tests)
├── main.py                        # Unified CLI entrypoint
├── requirements.txt               # Production Python dependencies
└── pytest.ini                     # Test configuration
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

```bash
# Clone the repository
git clone https://github.com/Anurag-amrev-7557/Pneumonia-Detection-DL.git
cd Pneumonia-Detection-DL

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download Pre-Trained Weights

Weights are automatically fetched on first web launch, or can be downloaded manually:

```bash
python scripts/download_weights.py models/current/
```

### 3. Launch the Clinical Workstation (Streamlit)

```bash
streamlit run src/ui/streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 💻 Unified CLI Reference (`main.py`)

```bash
# Single image prediction with Grad-CAM visualization
python main.py predict --image data/test/NORMAL/IM-0341-0001.jpeg --gradcam

# Single image prediction with margin cropping
python main.py predict --image path/to/scan.jpeg --crop-margins

# Batch directory evaluation
python main.py batch --input-dir data/test/NORMAL/ --output-file results.json

# Launch API server
python main.py serve --port 8000
```

---

## 🌐 REST API Service (FastAPI)

Launch the high-performance async API server:

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

* **Interactive Swagger UI**: `http://localhost:8000/docs`
* **Health Check**: `GET /health`
* **Prediction**: `POST /predict` (accepts multipart file upload)

---

## 🧪 Automated Test Suite

Run the full automated test suite verifying data pipelines, image processors, model factories, and inference engines:

```bash
pytest tests/ -v
```

All 28 tests pass with zero regressions.

---

## ⚖️ License & Medical Disclaimer

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

> [!IMPORTANT]
> **Medical Research Disclaimer**:
> PULMO·AI™ is developed as a computer vision research project and diagnostic decision support tool. It is not approved by the FDA or CE as a standalone diagnostic medical device. All outputs should be interpreted by a qualified radiologist or physician alongside patient history, clinical presentation, and laboratory findings.
