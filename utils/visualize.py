"""Visualization helpers for prototype segmentation results."""

import numpy as np
from PIL import Image


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple = (0, 255, 0),
    save_path: str | None = None,
) -> np.ndarray:
    """
    Blend a binary mask over an RGB image and optionally save it.

    Args:
        image: (H, W, 3) uint8 RGB array.
        mask: (H, W) uint8 binary array (1 = piece, 0 = background).
        alpha: opacity of the mask overlay in [0, 1].
        color: RGB colour used for the foreground region.
        save_path: if given, the result is saved as a PNG at this path.

    Returns:
        (H, W, 3) uint8 array with the mask blended in.
    """
    overlay = image.copy().astype(np.float32)
    color_layer = np.zeros_like(overlay)
    color_layer[mask > 0] = color

    fg = mask > 0
    overlay[fg] = (1.0 - alpha) * overlay[fg] + alpha * color_layer[fg]
    result = np.clip(overlay, 0, 255).astype(np.uint8)

    if save_path is not None:
        Image.fromarray(result).save(save_path)
        print(f"Saved overlay to {save_path}")

    return result


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Intersection-over-Union between two binary masks.

    Returns 1.0 when both masks are empty (true negative), 0.0 otherwise.
    """
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)
    intersection = int((pred & gt).sum())
    union        = int((pred | gt).sum())
    if union == 0:
        return 1.0
    return intersection / union
