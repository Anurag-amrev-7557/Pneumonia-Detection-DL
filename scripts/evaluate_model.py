#!/usr/bin/env python3
"""
Model evaluation script for comprehensive performance analysis.

Computes metrics like accuracy, sensitivity, specificity, precision, recall, F1, etc.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.models.inference import PneumoniaDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_on_directory(
    detector: PneumoniaDetector,
    test_dir: Path,
    class_mapping: dict | None = None
) -> dict:
    """
    Evaluate model on all images in a directory.
    
    Args:
        detector: PneumoniaDetector instance
        test_dir: Directory containing test images
        class_mapping: Mapping of directory names to labels
        
    Returns:
        dict: Evaluation metrics
    """
    if class_mapping is None:
        class_mapping = {'NORMAL': 0, 'PNEUMONIA': 1}
    
    y_true = []
    y_pred = []
    y_scores = []
    
    logger.info(f"Evaluating on {test_dir}")
    
    for class_name, true_label in class_mapping.items():
        class_dir = test_dir / class_name
        if not class_dir.exists():
            class_dir = test_dir / class_name.lower()
        
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue
        
        image_paths = []
        for ext in ('*.jpeg', '*.jpg', '*.png', '*.JPEG', '*.JPG', '*.PNG'):
            image_paths.extend(class_dir.glob(ext))
        logger.info(f"Found {len(image_paths)} {class_name} images")
        
        for img_path in image_paths:
            try:
                result = detector.predict_image(str(img_path))
                pred_label = 1 if result['is_pneumonia'] else 0
                confidence = result['probabilities']['Pneumonia']
                
                y_true.append(true_label)
                y_pred.append(pred_label)
                y_scores.append(confidence)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to process {img_path}: {e}")
    
    if not y_true:
        logger.error("No valid predictions made")
        return {}
    
    # Compute metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_scores)),
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics['confusion_matrix'] = {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
    }
    
    # Sensitivity and Specificity
    metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
    metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
    
    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Pneumonia'],
        output_dict=True
    )
    
    return metrics


def print_metrics(metrics: dict) -> None:
    """Print formatted metrics."""
    if not metrics:
        print("No metrics to display")
        return
    
    print("\n" + "="*70)
    print("MODEL EVALUATION METRICS")
    print("="*70)
    
    # Primary metrics
    print("\nPrimary Metrics:")
    print(f"  Accuracy:     {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision:    {metrics.get('precision', 0):.4f}")
    print(f"  Recall:       {metrics.get('recall', 0):.4f}")
    print(f"  F1 Score:     {metrics.get('f1_score', 0):.4f}")
    print(f"  AUC ROC:      {metrics.get('auc', 0):.4f}")
    
    # Medical metrics
    print("\nMedical Metrics:")
    print(f"  Sensitivity:  {metrics.get('sensitivity', 0):.4f}")
    print(f"  Specificity:  {metrics.get('specificity', 0):.4f}")
    
    # Confusion matrix
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        print("\nConfusion Matrix:")
        print(f"  True Negatives:  {cm['true_negatives']}")
        print(f"  False Positives: {cm['false_positives']}")
        print(f"  False Negatives: {cm['false_negatives']}")
        print(f"  True Positives:  {cm['true_positives']}")
    
    print("\n" + "="*70 + "\n")


def evaluate(
    test_dir: Path | str = Path('data/test'),
    model_path: str | Path | None = None,
    output: Path | str | None = None
) -> dict:
    """Run model evaluation and return metrics dictionary."""
    model_path = model_path or settings.inference.model_path
    test_dir = Path(test_dir)
    
    logger.info(f"Loading model from {model_path}")
    detector = PneumoniaDetector(model_path=model_path)
    
    logger.info(f"Evaluating on {test_dir}")
    metrics = evaluate_on_directory(detector, test_dir)
    
    if metrics:
        print_metrics(metrics)
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics saved to {output_path}")
    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate pneumonia detection model')
    parser.add_argument('--model', default=str(settings.inference.model_path), help='Path to trained model')
    parser.add_argument('--test-dir', type=Path, default=Path('data/test'), help='Path to test dataset')
    parser.add_argument('--output', type=Path, help='Save evaluation metrics to JSON file')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    
    args = parser.parse_args()
    try:
        metrics = evaluate(test_dir=args.test_dir, model_path=args.model, output=args.output)
        return 0 if metrics else 1
    except Exception:
        logger.exception("Evaluation failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
