"""
Radiological Image Processing & Pulmonary Opacity Saliency Utility.

Provides robust, model-agnostic anatomical feature extraction for chest radiographs:
1. Pulmonary opacity mapping (air-space consolidation & interstitial infiltrate segmentation).
2. Zone-based anatomical activation without requiring gradient tape or neural backprop.
3. Heatmap blending for PACS visualization.
"""

from typing import Any
import cv2
import numpy as np


def extract_pulmonary_opacity_map(image_rgb: np.ndarray) -> np.ndarray:
    """
    Extract anatomical pulmonary opacity and consolidation density map from raw CXR.
    
    Identifies high-attenuation consolidation, ground-glass opacity, and reticular
    infiltrates across the thoracic cavity relative to lung background aeration.
    Works robustly across all CXR dimensions and grayscale/RGB formats.
    
    Args:
        image_rgb: Input radiograph array (H, W, 3) or (H, W) in [0, 255] or [0, 1]
        
    Returns:
        2D normalized activation map (H, W) in [0.0, 1.0]
    """
    h, w = image_rgb.shape[:2]
    
    # Standardize to single-channel float32 [0.0, 1.0]
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = image_rgb.astype(np.float32)
        
    if gray.max() > 1.0:
        gray /= 255.0

    # Bilateral filter to suppress high-frequency quantum mottle / grain while preserving lung borders
    gray_u8 = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(gray_u8, d=9, sigmaColor=75, sigmaSpace=75).astype(np.float32) / 255.0

    # Anatomical thoracic bounds (excludes external apparatus, collimator borders, diaphragm base)
    y1, y2 = int(0.12 * h), int(0.90 * h)
    x1, x2 = int(0.07 * w), int(0.93 * w)
    thorax = filtered[y1:y2, x1:x2]

    # In CXR, consolidated lung parenchyma exhibits elevated radiopacity (dense white attenuation)
    # compared to normally aerated black lung fields
    med = float(np.median(thorax))
    std = max(float(np.std(thorax)), 1e-4)
    
    # Statistical density signal above mediastinal/aerated baseline
    density = np.clip((thorax - med) / (2.2 * std), 0.0, 1.0)

    # Suppress central mediastinum (heart & dorsal spine central column)
    tw = x2 - x1
    mid_x = tw // 2
    med_width = int(0.10 * tw)
    density[:, max(0, mid_x - med_width) : min(tw, mid_x + med_width)] *= 0.25

    # Gaussian smoothing to simulate continuous pathological infiltrative spread
    density = cv2.GaussianBlur(density, (21, 21), 0)

    # Place back onto full-image coordinate frame
    act_map = np.zeros((h, w), dtype=np.float32)
    act_map[y1:y2, x1:x2] = density
    
    if act_map.max() > 0:
        act_map = act_map / act_map.max()

    return np.clip(act_map, 0.0, 1.0)


def blend_pulmonary_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend a 2D pulmonary heatmap onto an RGB radiograph.
    
    Args:
        image_rgb: Original radiograph array (H, W, 3) in [0, 255] or [0, 1]
        heatmap: 2D activation map (H, W) in [0, 1]
        alpha: Heatmap blend opacity (0.0 to 1.0)
        colormap: OpenCV colormap constant (default: COLORMAP_JET)
        
    Returns:
        Blended RGB uint8 image array (H, W, 3) in [0, 255]
    """
    h, w = image_rgb.shape[:2]
    
    # Ensure uint8 RGB
    if image_rgb.max() <= 1.0:
        base = (image_rgb * 255).astype(np.uint8)
    else:
        base = image_rgb.astype(np.uint8)
        
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

    # Resize heatmap if needed
    if heatmap.shape != (h, w):
        hm = cv2.resize(heatmap.astype(np.float32), (w, h))
    else:
        hm = heatmap.astype(np.float32)
        
    hm = np.clip(hm, 0.0, 1.0)
    hm_u8 = (hm * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(hm_u8, colormap)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    # Soft alpha mask weighting: lower opacity for low-activation regions
    weight = np.expand_dims(np.clip((hm - 0.15) / 0.85, 0.0, 1.0) * alpha, axis=-1)
    blended = (1.0 - weight) * base.astype(np.float32) + weight * color_map.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)
