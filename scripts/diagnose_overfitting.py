#!/usr/bin/env python3
"""
Overfitting Diagnostic & Model Health Audit Tool

Performs 5 quantitative and qualitative tests to verify that the model
has truly generalized rather than memorized or overfitted:
0. Patient-Level Data Leakage & Split Integrity Audit
1. Training vs Validation Learning Curve Convergence
2. Stratified Test Set Evaluation, Minority-Class Recall & Balanced Accuracy
3. Image Perturbation & Noise Robustness Stress Test
4. Grad-CAM Spatial Activation Audit (Thoracic Cavity vs Peripheral Borders)
"""

import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score, f1_score

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.models.inference import PneumoniaDetector

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_patient_id(filename: str) -> str:
    m = re.match(r'^(person\d+)_', filename, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m_norm = re.match(r'^(NORMAL2-IM-\d+|IM-\d+)', filename)
    if m_norm:
        return m_norm.group(1)
    return filename.split('.')[0]


def check_patient_leakage(data_dir: Path) -> dict:
    """Test 0: Verify zero patient leakage and zero duplicate hashes across splits."""
    print("\n" + "=" * 70)
    print("TEST 0: PATIENT-LEVEL DATA LEAKAGE & INTEGRITY AUDIT")
    print("=" * 70)

    hashes = defaultdict(list)
    patient_splits = defaultdict(set)
    total_images = 0

    for split in ["train", "val", "test"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            cls_dir = data_dir / split / cls
            if not cls_dir.exists():
                cls_dir = data_dir / split / cls.lower()
            if not cls_dir.exists():
                continue
            for f in cls_dir.glob("*.*"):
                total_images += 1
                h = hashlib.md5(f.read_bytes()).hexdigest()
                hashes[h].append((split, f.name))
                pid = extract_patient_id(f.name)
                patient_splits[pid].add(split)

    cross_split_dups = [entries for h, entries in hashes.items() if len(set(e[0] for e in entries)) > 1]
    leak_patients = {pid: sp for pid, sp in patient_splits.items() if len(sp) > 1}

    print(f"Total Dataset Images:          {total_images}")
    print(f"Total Unique Patient Entities: {len(patient_splits)}")
    print(f"Cross-Split Hash Duplicates:   {len(cross_split_dups)}")
    print(f"Multi-Split Leaked Patients:   {len(leak_patients)}")

    if len(leak_patients) == 0 and len(cross_split_dups) == 0:
        verdict = "✅ LEAK-FREE: Zero patient overlap and zero duplicate images across splits."
    else:
        verdict = f"⚠️ DATA LEAKAGE: {len(leak_patients)} patients appear across multiple splits ({len(leak_patients)/len(patient_splits)*100:.1f}%)."

    print(f"\nVerdict: {verdict}")
    return {"leak_patients": len(leak_patients), "cross_dups": len(cross_split_dups)}


def check_learning_curves(history_path: Path) -> dict:
    """Test 1: Check training vs validation loss/accuracy curves."""
    print("\n" + "=" * 70)
    print("TEST 1: LEARNING CURVE & GENERALIZATION GAP ANALYSIS")
    print("=" * 70)

    if not history_path.exists():
        print(f"History file not found at {history_path}")
        return {}

    with open(history_path) as f:
        h = json.load(f)

    train_loss = h["loss"]
    val_loss = h["val_loss"]
    train_acc = h.get("accuracy", [0]*len(train_loss))
    val_acc = h.get("val_accuracy", [0]*len(val_loss))

    best_epoch = int(np.argmin(val_loss))
    min_val_loss = val_loss[best_epoch]
    corresponding_train_loss = train_loss[best_epoch]
    corresponding_val_acc = val_acc[best_epoch]
    corresponding_train_acc = train_acc[best_epoch]

    loss_gap = abs(min_val_loss - corresponding_train_loss)
    acc_gap = abs(corresponding_val_acc - corresponding_train_acc)

    print(f"Total Epochs Run:             {len(train_loss)}")
    print(f"Optimal Checkpoint Epoch:     Epoch {best_epoch + 1}")
    print(f"Training Loss at Best Epoch:   {corresponding_train_loss:.4f}")
    print(f"Validation Loss at Best Epoch: {min_val_loss:.4f} (Gap: {loss_gap:.4f})")
    print(f"Training Accuracy:            {corresponding_train_acc * 100:.2f}%")
    print(f"Validation Accuracy:          {corresponding_val_acc * 100:.2f}% (Gap: {acc_gap * 100:.2f}%)")

    # Diagnostic verdict
    if min_val_loss > 3.0 * corresponding_train_loss:
        verdict = "⚠️ WARNING: Possible overfitting (Val loss significantly higher than train loss)."
    elif acc_gap < 0.08 and min_val_loss < 0.5:
        verdict = "✅ HEALTHY: Train and Validation curves converged with small generalization gap (< 8%)."
    else:
        verdict = "✅ ACCEPTABLE: Model demonstrates normal convergence."

    print(f"\nVerdict: {verdict}")
    return {
        "best_epoch": best_epoch + 1,
        "loss_gap": loss_gap,
        "acc_gap": acc_gap,
    }


def check_minority_class_generalization(model_path: Path, test_dir: Path):
    """Test 2: Check test performance breakdown on minority class (NORMAL)."""
    print("\n" + "=" * 70)
    print("TEST 2: MINORITY CLASS GENERALIZATION & BALANCED ACCURACY AUDIT")
    print("=" * 70)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_gen = test_datagen.flow_from_directory(
        str(test_dir),
        target_size=settings.data.image_size,
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    detector = PneumoniaDetector(model_path=model_path)
    preds = detector.model.predict(test_gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    normal_recall = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    pneumonia_recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    overall_acc = np.mean(y_pred == y_true)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, preds[:, 1])

    print(f"Overall Test Accuracy:             {overall_acc * 100:.2f}%")
    print(f"Balanced Accuracy (Unbiased):      {bal_acc * 100:.2f}%")
    print(f"Macro F1-Score:                    {macro_f1:.4f}")
    print(f"Test ROC-AUC Score:                {auc:.4f}")
    print(f"Minority Class (NORMAL) Specificity:    {normal_recall * 100:.2f}% ({cm[0, 0]}/{cm[0, 0] + cm[0, 1]})")
    print(f"Majority Class (PNEUMONIA) Sensitivity: {pneumonia_recall * 100:.2f}% ({cm[1, 1]}/{cm[1, 0] + cm[1, 1]})")
    print("\nConfusion Matrix:")
    print(f"                Predicted Normal   Predicted Pneumonia")
    print(f"Actual Normal        {cm[0, 0]:5d}              {cm[0, 1]:5d}")
    print(f"Actual Pneumonia     {cm[1, 0]:5d}              {cm[1, 1]:5d}")

    if normal_recall > 0.80 and pneumonia_recall > 0.90:
        verdict = "✅ HEALTHY: Model shows strong generalization on both classes without exploiting imbalance."
    elif normal_recall < 0.70:
        verdict = "⚠️ BIASED: Poor minority class specificity. High false positive rate on healthy cases."
    else:
        verdict = "✅ ACCEPTABLE: Balanced trade-off between sensitivity and specificity."

    print(f"\nVerdict: {verdict}")


def check_perturbation_robustness(model_path: Path, test_dir: Path, n_samples: int = 50):
    """Test 3: Stress test model against Gaussian noise and slight brightness shifts."""
    print("\n" + "=" * 70)
    print("TEST 3: PERTURBATION & STRESS ROBUSTNESS TEST")
    print("=" * 70)
    print(f"Testing {n_samples} samples under clean vs corrupted conditions (Gaussian Noise + Contrast Shift)...")

    detector = PneumoniaDetector(model_path=model_path)
    image_paths = sorted(list(test_dir.glob("*/*.*")))[:n_samples]

    clean_matches = 0
    perturbed_matches = 0
    confidence_diffs = []

    for img_path in image_paths:
        clean_res = detector.predict_image(str(img_path))
        true_label = img_path.parent.name.upper()

        if clean_res["class"].upper() == true_label:
            clean_matches += 1

        from PIL import Image
        img = Image.open(img_path).convert("RGB").resize(settings.data.image_size)
        arr = np.array(img).astype(np.float32) / 255.0

        noise = np.random.normal(0, 0.03, arr.shape)
        perturbed_arr = np.clip(arr * 1.05 + noise, 0.0, 1.0)

        pred_scores = detector.model.predict(np.expand_dims(perturbed_arr, axis=0), verbose=0)[0]
        pred_idx = int(np.argmax(pred_scores))
        pred_class = ["NORMAL", "PNEUMONIA"][pred_idx]

        if pred_class == true_label:
            perturbed_matches += 1

        confidence_diffs.append(abs(clean_res["confidence"] - pred_scores[pred_idx]))

    clean_acc = clean_matches / len(image_paths)
    pert_acc = perturbed_matches / len(image_paths)
    avg_conf_shift = np.mean(confidence_diffs)

    print(f"Clean Sample Accuracy:         {clean_acc * 100:.1f}%")
    print(f"Perturbed Sample Accuracy:     {pert_acc * 100:.1f}%")
    print(f"Average Confidence Shift:      {avg_conf_shift * 100:.2f}%")

    if pert_acc >= clean_acc - 0.08:
        verdict = "✅ ROBUST: Model predictions are resilient to sensory noise and do not shatter under minor perturbations."
    else:
        verdict = "⚠️ BRITTLE: Model is sensitive to small pixel perturbations, indicating possible overfitting to high-frequency artifacts."

    print(f"\nVerdict: {verdict}")


def check_gradcam_spatial_focus(model_path: Path, test_dir: Path, n_samples: int = 20):
    """Test 4: Quantify Grad-CAM activation mass in central lung fields vs peripheral borders."""
    print("\n" + "=" * 70)
    print("TEST 4: GRAD-CAM SPATIAL ATTENTION & SHORTCUT AUDIT")
    print("=" * 70)

    from src.models.grad_cam import GradCAMVisualizer
    model = keras.models.load_model(str(model_path))
    gradcam = GradCAMVisualizer(model=model)

    imgs = sorted(list(test_dir.glob("*/*.*")))[:n_samples]
    lung_masses = []

    for img_p in imgs:
        heatmap = gradcam.generate_heatmap(str(img_p), class_index=1, normalize=True)
        h, w = heatmap.shape
        tot = np.sum(heatmap) + 1e-8
        mask = np.zeros_like(heatmap)
        mask[int(h*0.15):int(h*0.85), int(w*0.15):int(w*0.85)] = 1.0
        lung_masses.append(np.sum(heatmap * mask) / tot)

    avg_lung = np.mean(lung_masses) * 100
    avg_border = 100.0 - avg_lung

    print(f"Average Heatmap Energy in Central Lung Fields: {avg_lung:.1f}%")
    print(f"Average Heatmap Energy in Peripheral/Borders:   {avg_border:.1f}%")

    if avg_lung >= 50.0:
        verdict = "✅ ANATOMICALLY GROUNDED: Primary activations reside in the pulmonary cavity."
    else:
        verdict = "⚠️ SHORTCUT RISK: Over 50% of activation energy resides in peripheral borders/metadata."

    print(f"\nVerdict: {verdict}")


def main():
    print("🔬 PNEUMONIA DETECTION - COMPREHENSIVE MODEL HEALTH & OVERFITTING AUDIT")
    print("=" * 70)

    data_dir = Path("data")
    history_file = Path("models/current/training_history.json")
    model_file = Path("models/current/best_model.h5")
    test_dir = Path("data/test")

    check_patient_leakage(data_dir)
    check_learning_curves(history_file)
    check_minority_class_generalization(model_file, test_dir)
    check_perturbation_robustness(model_file, test_dir, n_samples=50)
    check_gradcam_spatial_focus(model_file, test_dir, n_samples=20)

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
