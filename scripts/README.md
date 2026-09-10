# Scripts Directory

Utility scripts and clinical audit tools for the Pneumonia Detection System.

## Available Diagnostic & Evaluation Tools

### 1. verify_zero_leakage.py
Performs an exhaustive cryptographic (MD5) duplicate audit and verifies that patient identifiers do not leak across Train, Val, and Test splits.

```bash
python scripts/verify_zero_leakage.py
```

**Checks**:
- Split image distribution
- Exact MD5 duplicates within and across splits
- Patient identifier cross-contamination (strictly 0.0% patient leakage)

---

### 2. diagnose_overfitting.py
Performs 5 quantitative and qualitative stress tests to confirm model generalization rather than background artifact memorization.

```bash
python scripts/diagnose_overfitting.py
```

**Diagnostic Tests**:
- Training vs validation learning curve convergence
- Stratified test set evaluation, minority-class recall & balanced accuracy
- Image perturbation & noise robustness stress test
- Grad-CAM spatial activation audit (thoracic cavity focus vs peripheral borders)

---

### 3. evaluate_model.py
Comprehensive model evaluation on the test set.

```bash
# Evaluate default model on test set
python scripts/evaluate_model.py --test-dir data/test

# Evaluate specific model checkpoint
python scripts/evaluate_model.py --model models/current/best_model.h5 --test-dir data/test

# Save metrics to JSON file
python scripts/evaluate_model.py --test-dir data/test --output evaluation_metrics.json
```

**Metrics Computed**:
- Accuracy, Precision, Recall, F1 Score
- Sensitivity, Specificity
- AUC-ROC
- Confusion Matrix & Classification Report

---

**Last Updated**: 2026-09-10
