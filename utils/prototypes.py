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


def _extract_features(dino_encoder, image_tensor: torch.Tensor, dino_num_layers: int) -> np.ndarray:
    """
    Run DINO on a single preprocessed image tensor.

    Args:
        image_tensor: (1, 3, H, W) float32 torch.Tensor, already on the correct device.
        dino_num_layers: number of transformer layers (n parameter for get_intermediate_layers).

    Returns:
        feats: (Hp, Wp, D) float32 numpy array, L2-normalized per patch.
    """
    with torch.no_grad():
        out = dino_encoder.get_intermediate_layers(
            image_tensor,
            n=range(dino_num_layers),
            reshape=True,
            norm=True,
        )[-1]  # (1, D, Hp, Wp)

    feats = out.squeeze(0).permute(1, 2, 0).float().cpu().numpy()  # (Hp, Wp, D)
    return _l2_norm(feats)


def _downsample_mask(mask: np.ndarray, Hp: int, Wp: int) -> np.ndarray:
    """
    Resize a binary mask (H, W) to the patch grid (Hp, Wp).
    A patch is labelled 'piece' when more than 50 % of its pixels are piece.
    Uses INTER_AREA for correct area-averaging then thresholds at 0.5.
    """
    mask_f = mask.astype(np.float32)
    resized = cv2.resize(mask_f, (Wp, Hp), interpolation=cv2.INTER_AREA)
    return (resized > 0.5).astype(np.uint8)


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_prototypes(
    support_set: list,
    dino_encoder,
    dino_num_layers: int,
    device,
    k: int = 5,
) -> dict:
    """
    Build multi-class prototypes from an annotated support set.

    Args:
        support_set: list of (image_tensor, mask_np) where
            image_tensor is (1, 3, H, W) float32 on *device*,
            mask_np is (H_orig, W_orig) uint8 with 1=piece, 0=background.
        dino_encoder: loaded DINOv3 model (eval mode).
        dino_num_layers: number of layers to pass to get_intermediate_layers.
        device: torch.device.
        k: number of prototypes per class (reduced if not enough samples).

    Returns:
        {"piece": (Kp, D) float32, "background": (Kb, D) float32}
    """
    piece_pool, bg_pool = [], []

    for image_tensor, mask_np in support_set:
        feats = _extract_features(dino_encoder, image_tensor, dino_num_layers)  # (Hp, Wp, D)
        Hp, Wp, _ = feats.shape
        patch_mask = _downsample_mask(mask_np, Hp, Wp)  # (Hp, Wp)

        piece_vecs = feats[patch_mask == 1]
        bg_vecs    = feats[patch_mask == 0]

        if len(piece_vecs):
            piece_pool.append(piece_vecs)
        if len(bg_vecs):
            bg_pool.append(bg_vecs)

    if not piece_pool:
        raise ValueError("No foreground patches found in support set — check your masks.")
    if not bg_pool:
        raise ValueError("No background patches found in support set — check your masks.")

    piece_vecs = np.concatenate(piece_pool, axis=0)  # (N, D)
    bg_vecs    = np.concatenate(bg_pool,    axis=0)  # (M, D)

    return {
        "piece":      _cluster(piece_vecs, k),  # (Kp, D)
        "background": _cluster(bg_vecs,    k),  # (Kb, D)
    }


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
) -> tuple:
    """
    Segment a query image using prototype cosine matching.

    Args:
        image_tensor: (1, 3, H, W) float32 torch.Tensor on *device*.
        protos: dict from build_prototypes or load_prototypes.
        dino_encoder: loaded DINOv3 model (eval mode).
        dino_num_layers: number of layers passed to get_intermediate_layers.
        device: torch.device.
        tau: confidence threshold on sim_piece (default 0.5).
        min_area: minimum connected-component area in pixels (at image resolution).
        morph_kernel: side of the elliptic morphological structuring element.

    Returns:
        mask: (H, W) uint8 binary mask at image resolution (H, W same as image_tensor).
        score_map: (Hp, Wp) float32 patch-resolution score map (sim_piece – sim_bg).
    """
    feats = _extract_features(dino_encoder, image_tensor, dino_num_layers)  # (Hp, Wp, D)
    Hp, Wp, D = feats.shape
    H_img = image_tensor.shape[-2]
    W_img = image_tensor.shape[-1]

    flat = feats.reshape(-1, D).astype(np.float32)  # (N, D)

    piece_protos = protos["piece"].astype(np.float32)       # (Kp, D)
    bg_protos    = protos["background"].astype(np.float32)  # (Kb, D)

    # Cosine similarity = dot product (all vectors already L2-normalized)
    sim_piece = (flat @ piece_protos.T).max(axis=1)  # (N,) – best prototype per patch
    sim_bg    = (flat @ bg_protos.T).max(axis=1)     # (N,)

    score_map     = (sim_piece - sim_bg).reshape(Hp, Wp).astype(np.float32)
    sim_piece_map = sim_piece.reshape(Hp, Wp).astype(np.float32)

    # Bilinear upsample both maps to image resolution
    def _upsample(arr: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(arr)[None, None]  # (1, 1, Hp, Wp)
        up = F.interpolate(t, size=(H_img, W_img), mode="bilinear", align_corners=False)
        return up.squeeze().numpy()

    score_up     = _upsample(score_map)      # (H, W)
    sim_piece_up = _upsample(sim_piece_map)  # (H, W)

    # Threshold: piece if it beats background AND exceeds confidence
    raw_mask = ((score_up > 0) & (sim_piece_up > tau)).astype(np.uint8)

    # Morphological cleanup: open (remove noise) then close (fill holes)
    if morph_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )
        cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  kernel)
        cleaned = cv2.morphologyEx(cleaned,  cv2.MORPH_CLOSE, kernel)
    else:
        cleaned = raw_mask

    # Discard small connected components
    final_mask = np.zeros_like(cleaned, dtype=np.uint8)
    if min_area > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                final_mask[labels == lbl] = 1
    else:
        final_mask = cleaned

    return final_mask, score_map
