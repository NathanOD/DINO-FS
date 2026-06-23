"""Visualization helpers for prototype segmentation results."""

import numpy as np
from PIL import Image

# Default color palette for foreground classes (RGB)
_CLASS_COLORS = [
    (0,   200,  50),   # base    — green
    (220,  80,  20),   # welded  — orange
    (50,  120, 220),   # class 3 — blue
    (200,  30, 200),   # class 4 — purple
]


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


def overlay_label_map(
    image: np.ndarray,
    label_map: np.ndarray,
    class_names: list[str],
    alpha: float = 0.5,
    save_path: str | None = None,
) -> np.ndarray:
    """
    Blend a multi-class label map over an RGB image.

    Args:
        image: (H, W, 3) uint8 RGB array.
        label_map: (H, W) uint8 with 0=background and i+1=class_names[i].
        class_names: ordered list of foreground class names.
        alpha: opacity of the colour overlay.
        save_path: if given, saved as PNG.

    Returns:
        (H, W, 3) uint8 blended image.
    """
    overlay = image.copy().astype(np.float32)
    for j, name in enumerate(class_names):
        color = _CLASS_COLORS[j % len(_CLASS_COLORS)]
        fg = label_map == j + 1
        if not fg.any():
            continue
        overlay[fg] = (1.0 - alpha) * overlay[fg] + alpha * np.array(color, dtype=np.float32)

    result = np.clip(overlay, 0, 255).astype(np.uint8)
    if save_path is not None:
        Image.fromarray(result).save(save_path)
        print(f"Saved overlay to {save_path}")
    return result


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection-over-Union between two binary masks."""
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)
    intersection = int((pred & gt).sum())
    union        = int((pred | gt).sum())
    if union == 0:
        return 1.0
    return intersection / union


def compute_mean_iou(
    pred_label_map: np.ndarray,
    gt_label_map: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    """
    Per-class IoU between two integer label maps.

    Returns a dict {class_name: iou} for each foreground class.
    """
    ious: dict[str, float] = {}
    for j, name in enumerate(class_names):
        label = j + 1
        pred = pred_label_map == label
        gt   = gt_label_map   == label
        intersection = int((pred & gt).sum())
        union        = int((pred | gt).sum())
        ious[name] = 1.0 if union == 0 else intersection / union
    return ious
