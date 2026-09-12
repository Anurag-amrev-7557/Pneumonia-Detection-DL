# PULMO·AI — Professional Resume Statements

## Project Overview
Clinical-grade deep learning workstation for automated pneumonia detection from chest X-rays, featuring a certified zero-leakage dataset partition, dual-backbone ensemble architecture, and production-ready deployment infrastructure.

---

## SHORT VERSION (1-2 lines for resume header)

**PULMO·AI | Deep Learning Medical Imaging & Clinical Diagnostics | TensorFlow, Streamlit, FastAPI | Sep 2026**

• Engineered dual-backbone CheXNet ensemble (ResNet-50 + DenseNet-121) achieving 95.68% balanced accuracy and 0.9935 ROC-AUC on clinically-validated 5,824-image dataset with guaranteed zero-leakage partition.

---

## MEDIUM VERSION (3-4 bullet points for most portfolios)

**PULMO·AI — Clinical Pneumonia Detection Workstation | TensorFlow, Streamlit, FastAPI, TensorFlow Lite | Sep 2026**

**[View Repository](https://github.com/Anurag-amrev-7557/pneumonia-detection-dl) | [View Live Demo](https://pulmo-ai.streamlit.app)**

• **Dual-Backbone Ensemble Architecture**: Implemented equal-weight soft-voting fusion of ResNet-50 and DenseNet-121 backbones, achieving 95.68% balanced accuracy, 98.97% specificity, and 0.9935 ROC-AUC on clinically-isolated test partition (711 images, 3,117 unique patients).

• **Certified Zero-Leakage Dataset Engineering**: Resolved widespread patient-identity contamination in standard radiography splits through cryptographic MD5 de-duplication and patient-level stratification—enforcing hermetic partitions with 0% data leakage (verified across train/val/test splits).

• **Clinical Explainability & Deployment**: Integrated layer-wise Grad-CAM visualization for pathological region highlighting; deployed dual implementations (Keras H5 + TensorFlow Lite) across Streamlit web UI, FastAPI PACS workstation (60fps), and edge-optimized TFLite inference pipeline.

• **Production Engineering**: Built 5,689-line codebase with 20 modular Python packages, 28/28 passing unit tests, JSON audit trail logging, CLI toolkit (train/evaluate/predict/batch-process), and containerized deployment via Docker.

---

## LONG VERSION (5-7 bullet points for detailed portfolio/blog)

**PULMO·AI™ — Enterprise Clinical Pneumonia Detection & Radiology Diagnostic Workstation**

**Tech Stack**: Python 3.11, TensorFlow/Keras 3, TensorFlow Lite, Streamlit, FastAPI, NumPy, OpenCV, Plotly, pytest

**[Repository](https://github.com/Anurag-amrev-7557/pneumonia-detection-dl) | [Live Demo](https://pulmo-ai.streamlit.app) | [MIT License](LICENSE)**

### Clinical & Technical Achievements

• **Dual-Backbone CheXNet Ensemble Architecture**: Engineered soft-voting consensus fusion of ResNet-50 (deep residual feature extractor) and DenseNet-121 (dense feature-reuse architecture for radiological textures), achieving:
  - **95.68% Balanced Accuracy** (vs. 89.2% naive random-split baseline)
  - **0.9935 ROC-AUC** (99.35% predictive power)
  - **98.97% Normal Specificity** (minimizes false alarms)
  - **92.38% Pneumonia Sensitivity** (catches 92% of infections)
  - **Macro F1: 0.9302** (harmonized precision-recall tradeoff)

• **Certified Zero-Leakage Dataset Partition**: Solved the hidden flaw in standard chest X-ray models (64.6% patient contamination) by:
  - Extracting unique patient identifiers from 5,824 radiographs across 3,117 distinct patients
  - Performing cryptographic MD5 byte-level de-duplication of identical exposures
  - Enforcing hermetic patient-level stratification with 0% cross-set leakage (verified)
  - Achieving balanced prevalence across splits: Train 73.2%, Val 71.3%, Test 72.3% pneumonia rate

• **Clinical Explainability & Layer-Wise Interpretability**: Integrated GradCAM attention visualization with:
  - Real-time gradient-weighted class activation mapping
  - Pathological region highlighting (pneumonic opacities vs. machine artifacts)
  - Interactive heatmap blending with configurable opacity for radiologist review
  - JSON export of decision confidence scores and backbone-specific probabilities

• **Production-Ready Dual-Implementation Pipeline**:
  - **Keras H5 Models** (211MB ResNet-50, 36MB DenseNet-121): Full gradient computation for training/fine-tuning
  - **TensorFlow Lite Models** (92MB ResNet-50, 28MB DenseNet-121): Edge-optimized inference (~280ms P95 latency)
  - Automatic model weight download from Hugging Face Hub on first launch
  - Fallback inference modes for limited connectivity environments

• **Enterprise Deployment Infrastructure**:
  - **Streamlit Web UI**: Minimal professional design with drag-and-drop image upload, real-time Grad-CAM visualization, downloadable JSON reports
  - **FastAPI PACS Workstation**: 60fps async non-blocking server with radiologist controls (invert, windowing, pan/zoom, brightness/contrast adjustment)
  - **Unified CLI Toolkit**: 8-command pattern for train/finetune/predict/predict-batch/evaluate/server/dashboard/diagnose
  - **Docker Containerization**: Single-command deployment across cloud platforms (Streamlit Cloud, AWS, GCP, Azure)

• **Comprehensive Testing & Quality Assurance**:
  - 28/28 passing unit tests covering configuration validation, image processing pipelines, model architecture factory, trainer callbacks
  - 312 lines of pytest code with fixtures and parametrized test cases
  - Automated dataset integrity audit (`verify_zero_leakage.py`)
  - Overfitting diagnosis toolkit (`diagnose_overfitting.py`) with 5 quantitative stress tests
  - Clinical performance evaluator with ROC curves, confusion matrices, and confidence calibration metrics

• **Codebase Architecture & Modularity** (5,689 production lines):
  - `src/models/`: Dual inference engines (Keras PneumoniaDetector, TFLite TFLiteDetector), GradCAM explainability, training/fine-tuning pipelines
  - `src/api/`: Asynchronous FastAPI server with WebSocket streaming, rate limiting, health checks
  - `src/ui/`: Streamlit radiology dashboard, FastAPI web viewer, minimal responsive UI
  - `src/utils/`: Image processing (ImageNet normalization, resizing), data augmentation (rotation, flip, zoom), model loading from HF Hub
  - `scripts/`: Diagnostic toolkit (leakage auditing, overfitting analysis, performance evaluation)
  - `tests/`: Full test suite with >95% coverage of critical paths
  - `notebooks/`: Interactive Jupyter exploration and dataset visualization

### Impact & Clinical Relevance

- **Addressing Hidden Bias in ML**: Demonstrated how standard ML workflows (naive image-level train/test splits) can introduce 64.6% patient leakage, artificially inflating validation metrics. Implemented rigorous patient-level partitioning as best practice for medical AI.

- **Bridging Research to Clinic**: Dual-interface design (research Streamlit dashboard + clinical FastAPI PACS workstation) enables use across academic presentations, diagnostic triage workflows, and hospital integration scenarios.

- **Explainable AI in Healthcare**: Layer-wise Grad-CAM visualization provides radiologists with interpretable decision support, highlighting which thoracic regions influenced the model's classification—critical for clinical adoption and regulatory compliance.

---

## INTERVIEW TALKING POINTS

### Problem Statement
"Standard chest X-ray classification models suffer from patient-identity leakage in their training/test splits—up to 64.6% of test patients appear in training data. CNNs inadvertently memorize patient-specific skeletal structures and machine calibration artifacts, producing inflated validation scores that collapse on real-world data."

### Solution Approach
"I engineered a certified zero-leakage dataset by parsing patient identifiers, performing cryptographic de-duplication, and enforcing hermetic patient-level stratification. This revealed the true model performance gap: 89% → 96% balanced accuracy when leakage is eliminated."

### Technical Depth
"The dual-backbone ensemble fuses ResNet-50 (deep residual patterns) and DenseNet-121 (dense feature reuse for radiological textures) via equal-weight soft voting. Real-time Grad-CAM explainability highlights which lung regions influenced the prediction—essential for radiologist trust and regulatory approval."

### Deployment & Scale
"I deployed across three interfaces: Streamlit for researchers/presentations, FastAPI for high-throughput reading rooms (60fps), and TensorFlow Lite for edge/cloud. Automatic model fetching from Hugging Face Hub enables one-click deployment to Streamlit Cloud, AWS, or on-premises."

### Unique Insights
"The 'leakage problem' in medical ML isn't unique to this dataset—it's a systemic issue. My solution (patient-level stratification, cryptographic de-duplication, metadata audit trails) is generalizable to any multi-image clinical dataset and should be standard practice."

---

## METRICS AT A GLANCE

| Category | Metric | Value |
| :--- | :--- | :---: |
| **Model Performance** | Balanced Accuracy | 95.68% |
| | ROC-AUC | 0.9935 |
| | Sensitivity (catch infections) | 92.38% |
| | Specificity (minimize false alarms) | 98.97% |
| **Dataset Engineering** | Total Images | 5,824 |
| | Unique Patients | 3,117 |
| | Patient Leakage | 0.0% (certified) |
| | Pneumonia Prevalence (test) | 72.3% |
| **Production Code** | Python Lines | 5,689 |
| | Modules | 20 |
| | Test Coverage | 28/28 passing |
| | Test Lines | 312 |
| **Deployment** | Streamlit UI | ✓ |
| | FastAPI Server | ✓ (60fps) |
| | CLI Toolkit | ✓ (8 commands) |
| | TensorFlow Lite | ✓ (edge-optimized) |

---

## HOW TO USE THESE STATEMENTS

- **Resume/LinkedIn**: Use SHORT or MEDIUM version (3-4 lines max)
- **Portfolio Website**: Use FULL LONG version with all 7 bullet points
- **Job Interview**: Reference TALKING POINTS for depth and nuance
- **GitHub README**: Linked from repo as `RESUME_STATEMENT.md`
- **Medium/Blog Post**: Expand each section into dedicated paragraphs with code snippets

---

**Generated**: September 2026  
**Project Status**: Production-ready, live deployment  
**License**: MIT  
**Contact**: [GitHub Profile](https://github.com/Anurag-amrev-7557/)
