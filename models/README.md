# Models Directory Structure & Diagnostic Weights

This directory contains production-certified deep learning model weights, architecture telemetry, ensemble configuration, and training performance history for **PULMO·AI™**.

---

## 🗂️ Active Production Models (`models/current/`)

| File Name | Architecture | Framework | Size | Role & Clinical Purpose |
| :--- | :--- | :--- | :---: | :--- |
| **`best_model.h5`** | ResNet-50 | Keras / TF | ~211 MB | Primary deep residual feature extractor |
| **`densenet121_best.h5`** | DenseNet-121 | Keras / TF | ~36 MB | CheXNet dense feature reuse partner |
| **`best_model.tflite`** | ResNet-50 (TFLite) | LiteRT | ~23 MB | Quantized edge / Streamlit Cloud runtime |
| **`densenet121_best.tflite`** | DenseNet-121 (TFLite) | LiteRT | ~7 MB | Quantized edge / Streamlit Cloud runtime |
| **`ensemble_metadata.json`** | Soft-Voting Dual Ensemble | JSON | < 1 KB | Verified consensus weights & test set metrics |
| **`model_metadata.json`** | ResNet-50 Metadata | JSON | < 1 KB | Training hyperparameters and class weight ratios |
| **`training_history.json`** | Optimization Log | JSON | ~3 KB | Loss, accuracy, AUC curves across epochs |

---

## 🔬 Benchmark Performance (Certified Test Cohort)

Metrics evaluated on the held-out test cohort:

| Metric | Primary ResNet-50 | DenseNet-121 (CheXNet) | **CheXNet Dual Ensemble (50/50)** |
| :--- | :---: | :---: | :---: |
| **Balanced Accuracy** | 93.46% | 96.16% | **95.68%** |
| **ROC-AUC** | 0.9879 | 0.9972 | **0.9935** |
| **Macro F1-Score** | 0.9074 | 0.9442 | **0.9302** |
| **Specificity (Normal)** | 96.45% | 97.97% | **98.97%** *(Zero false alarm focus)* |
| **Sensitivity (Pneumonia)** | 90.47% | 94.36% | **92.38%** |

### Ensemble Mechanism
The production engine implements equal-weight soft voting between the two architecturally divergent backbones:
$$P_{\text{final}}(\text{Pneumonia}) = 0.5 \cdot P_{\text{ResNet-50}}(\text{Pneumonia}) + 0.5 \cdot P_{\text{DenseNet-121}}(\text{Pneumonia})$$

* **High Specificity:** DenseNet-121's dense connectivity suppresses false positive alarms on normal bronchovascular markings, achieving **98.97% specificity on normal radiographs**.
* **Clinical Interpretability:** Grad-CAM saliency heatmaps visualize attention on the final convolutional layer (`conv5_block16_2_conv` / `conv5_block3_out`), localizing consolidations and infiltrates.

---

## 📥 Automated Weight Download

For local environments where weights are not pre-packaged:

```bash
# Automated download via GitHub Releases with SHA-256 verification
python scripts/download_weights.py models/current/
```

Or when running the web application, weights are automatically retrieved from Hugging Face Hub (`Anurag234/pulmo-ai-weights`) on first execution.
