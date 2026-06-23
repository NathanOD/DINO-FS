"""
Few-shot prototype segmentation using DINOv2 dense features.

Pipeline:
  1. build_prototypes  – offline, once per support set
  2. segment           – per query image, no gradient
"""

import pickle
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _l2_norm(x: np.ndarray) -> np.ndarray:
    """L2-normalize along the last axis (in-place safe copy)."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def _extract_features(
    dino_encoder,
    image_tensor: torch.Tensor,
    dino_num_layers: int,
    naf_model=None,
    naf_scale: int = 1,
) -> np.ndarray:
    """
    Run DINO on a single preprocessed image tensor, optionally upsampled with NAF.

    Args:
        image_tensor: (1, 3, H, W) float32 torch.Tensor, already on the correct device.
        dino_num_layers: number of transformer layers (n parameter for get_intermediate_layers).
        naf_model: optional NAF upsampler. If provided, features are upsampled by naf_scale.
        naf_scale: spatial upsampling factor relative to the DINO patch grid (default 1 = no NAF).

    Returns:
        feats: (Hp*naf_scale, Wp*naf_scale, D) float32 numpy array, L2-normalized per patch.
    """
    with torch.no_grad():
        out = dino_encoder.get_intermediate_layers(
            image_tensor,
            n=range(dino_num_layers),
            reshape=True,
            norm=True,
        )[-1]  # (1, D, Hp, Wp)

        if naf_model is not None and naf_scale > 1:
            Hp, Wp = out.shape[-2:]
            out = naf_model(image_tensor, out, (Hp * naf_scale, Wp * naf_scale))  # (1, D, Hp*s, Wp*s)

    feats = out.squeeze(0).permute(1, 2, 0).float().cpu().numpy()  # (Hp', Wp', D)
    return _l2_norm(feats)


def _downsample_label_map(label_map: np.ndarray, Hp: int, Wp: int) -> np.ndarray:
    """Resize an integer label map (H, W) to the patch grid (Hp, Wp) using nearest-neighbor."""
    return cv2.resize(label_map, (Wp, Hp), interpolation=cv2.INTER_NEAREST)


def _cluster(vecs: np.ndarray, k: int) -> np.ndarray:
    """
    K-means clustering on a set of feature vectors.
    Reduces k if fewer samples than k are available.
    Returns L2-normalized centroids (k', D).
    """
    actual_k = min(k, len(vecs))
    if actual_k == 1:
        return _l2_norm(vecs.mean(axis=0, keepdims=True))
    km = KMeans(n_clusters=actual_k, n_init=10, random_state=0)
    km.fit(vecs)
    return _l2_norm(km.cluster_centers_.astype(np.float32))


def _morph_cleanup(mask: np.ndarray, kernel_size: int, min_area: int) -> np.ndarray:
    """Open + close a binary mask, then discard small connected components."""
    if kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if min_area > 0:
        out = np.zeros_like(mask)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                out[labels == lbl] = 1
        return out
    return mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_prototypes(
    support_set: list,
    dino_encoder,
    dino_num_layers: int,
    device,
    k: int = 5,
    class_names: list[str] = ("piece",),
    naf_model=None,
    naf_scale: int = 1,
) -> dict:
    """
    Build multi-class prototypes from an annotated support set.

    Args:
        support_set: list of (image_tensor, label_map) where
            image_tensor is (1, 3, H, W) float32 on *device*,
            label_map is (H_orig, W_orig) uint8 with 0=background and i+1=class_names[i].
        dino_encoder: loaded DINOv3 model (eval mode).
        dino_num_layers: number of layers to pass to get_intermediate_layers.
        device: torch.device.
        k: number of prototypes per class (reduced if not enough samples).
        class_names: ordered list of foreground class names, e.g. ["base", "welded"].
            Label 1 maps to class_names[0], label 2 to class_names[1], etc.

    Returns:
        dict with one (K, D) array per class and "background", plus a merged "foreground"
        prototype (all fg classes combined) and metadata keys "class_names" and "naf_scale".
    """
    pools = {name: [] for name in class_names}
    pools["background"] = []
    fg_pool: list = []

    for i, (image_tensor, label_map) in enumerate(support_set):
        feats = _extract_features(dino_encoder, image_tensor, dino_num_layers, naf_model, naf_scale)
        Hp, Wp, _ = feats.shape
        patch_labels = _downsample_label_map(label_map, Hp, Wp)  # (Hp, Wp) uint8

        fg_vecs = feats[patch_labels > 0]
        if len(fg_vecs):
            fg_pool.append(fg_vecs)

        for j, name in enumerate(class_names):
            vecs = feats[patch_labels == j + 1]
            if len(vecs):
                pools[name].append(vecs)

        bg_vecs = feats[patch_labels == 0]
        if len(bg_vecs):
            pools["background"].append(bg_vecs)

    protos: dict = {}
    for name, pool in pools.items():
        if not pool:
            raise ValueError(f"No patches found for class '{name}' — check your masks.")
        protos[name] = _cluster(np.concatenate(pool, axis=0), k)

    protos["foreground"]  = _cluster(np.concatenate(fg_pool, axis=0), k)
    protos["class_names"] = list(class_names)
    protos["naf_scale"]   = naf_scale
    return protos


def save_prototypes(protos: dict, path: str) -> None:
    """Persist prototypes dict to disk (pickle)."""
    with open(path, "wb") as f:
        pickle.dump(protos, f)


def load_prototypes(path: str) -> dict:
    """Load prototypes dict from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def segment(
    image_tensor: torch.Tensor,
    protos: dict,
    dino_encoder,
    dino_num_layers: int,
    device,
    tau: float = 0.5,
    min_area: int = 200,
    morph_kernel: int = 5,
    naf_model=None,
) -> tuple:
    """
    Segment a query image using prototype cosine matching.

    Args:
        image_tensor: (1, 3, H, W) float32 torch.Tensor on *device*.
        protos: dict from build_prototypes or load_prototypes.
        dino_encoder: loaded DINOv3 model (eval mode).
        dino_num_layers: number of layers passed to get_intermediate_layers.
        device: torch.device.
        tau: minimum absolute similarity for a foreground class to win over background.
        min_area: minimum connected-component area in pixels (at image resolution).
        morph_kernel: side of the elliptic morphological structuring element.

    Returns:
        label_map: (H, W) uint8 — 0=background, i+1=class_names[i].
        score_maps: dict {class_name: (Hp, Wp) float32} of sim(class) − sim(background).
    """
    class_names = protos.get("class_names", ["piece"])
    naf_scale   = protos.get("naf_scale", 1)

    H_img = image_tensor.shape[-2]
    W_img = image_tensor.shape[-1]

    feats = _extract_features(dino_encoder, image_tensor, dino_num_layers, naf_model, naf_scale)
    Hp, Wp, D = feats.shape
    flat = feats.reshape(-1, D).astype(np.float32)  # (N, D)

    bg_protos = protos["background"].astype(np.float32)
    sim_bg = (flat @ bg_protos.T).max(axis=1)  # (N,)

    # --- Stage 1: piece vs background ---
    piece_mask: np.ndarray | None = None
    if "foreground" in protos:
        sim_fg = (flat @ protos["foreground"].astype(np.float32).T).max(axis=1)
        piece_mask = sim_fg >= sim_bg

    # --- Stage 2: per-class discrimination ---
    sim_classes: dict[str, np.ndarray] = {}
    score_maps: dict[str, np.ndarray] = {}
    for name in class_names:
        cls_protos = protos[name].astype(np.float32)
        sim = (flat @ cls_protos.T).max(axis=1)  # (N,)
        sim_classes[name] = sim
        score_maps[name] = (sim - sim_bg).reshape(Hp, Wp).astype(np.float32)

    fg_sim_stack = np.stack([sim_classes[n] for n in class_names], axis=1)  # (N, C)

    if piece_mask is not None:
        # Two-stage: stage 1 decided piece/bg, stage 2 picks best foreground class within piece.
        fg_best = fg_sim_stack.argmax(axis=1) + 1  # (N,) — 1=base, 2=welded, ...
        raw_labels = np.where(piece_mask, fg_best, 0).astype(np.uint8)
    else:
        # Single-stage fallback: winner-takes-all including background
        sim_stack = np.stack([sim_bg] + [sim_classes[n] for n in class_names], axis=1)
        raw_labels = sim_stack.argmax(axis=1)
        win_sim = sim_stack[np.arange(len(raw_labels)), raw_labels]
        raw_labels[(raw_labels > 0) & (win_sim < tau)] = 0
        raw_labels = raw_labels.astype(np.uint8)

    # Upsample label map to image resolution (nearest-neighbor preserves class ids)
    raw_label_map = raw_labels.reshape(Hp, Wp).astype(np.uint8)
    label_map_up = F.interpolate(
        torch.from_numpy(raw_label_map).float()[None, None],
        size=(H_img, W_img),
        mode="nearest",
    ).squeeze().byte().numpy()

    # Morphological cleanup per class; background is whatever is unclaimed
    final_label_map = np.zeros((H_img, W_img), dtype=np.uint8)
    for j, name in enumerate(class_names):
        cls_mask = (label_map_up == j + 1).astype(np.uint8)
        cls_mask = _morph_cleanup(cls_mask, morph_kernel, min_area)
        final_label_map[cls_mask > 0] = j + 1

    return final_label_map, score_maps
