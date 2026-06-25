"""
Depth geometry utilities for RGB-D few-shot segmentation.

Pipeline per image:
  1. load_depth_mm          – load uint16 depth PNG → float32 mm
  2. load_pose              – parse robot end-effector pose T_bg from txt file
  3. load_calibration       – load T_gc (hand-eye) and camera intrinsics
  4. depth_to_base_pointcloud  – project depth to 3D in robot base frame
                                 P_base = T_bg @ T_gc @ P_camera
  5. compute_patch_geo_features – aggregate per-patch: XYZ centroid (3),
                                  surface normal (3), depth std (1) → (fH, fW, 7)
"""

import json
import numpy as np
import cv2
import yaml


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_depth_mm(path: str, target_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Load a uint16 depth PNG and return a float32 array in mm.

    Args:
        path: path to the depth PNG.
        target_hw: optional (H, W) to resize with INTER_NEAREST.

    Returns:
        (H, W) float32 in mm. Zero where no depth reading.
    """
    depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    if target_hw is not None and depth.shape[:2] != target_hw:
        depth = cv2.resize(depth, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return depth


def load_pose(path: str) -> np.ndarray:
    """Parse a 4×4 robot pose matrix from a text file.

    Expected format (MATLAB-style): '[r00, r01, ..., tx ; ... ; 0,0,0,1]'
    Uses regex extraction to be robust to trailing brackets/semicolons.

    Returns:
        T_bg: (4, 4) float64, gripper-to-base transform (robot FK at capture time).
    """
    import re
    with open(path) as f:
        text = f.read()
    numbers = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]
    return np.array(numbers, dtype=np.float64).reshape(4, 4)


def load_calibration(handeye_path: str, intrinsics_path: str) -> tuple[np.ndarray, dict]:
    """Load hand-eye calibration and camera intrinsics.

    Args:
        handeye_path:   YAML or JSON file containing 'T_gc' key.
        intrinsics_path: YAML file with fx, fy, cx, cy at original sensor resolution.

    Returns:
        T_gc:      (4, 4) float64 — camera-to-gripper static transform.
        intrinsics: dict with keys fx, fy, cx, cy (pixels, original resolution).
    """
    if handeye_path.endswith(".json"):
        with open(handeye_path) as f:
            T_gc = np.array(json.load(f)["T_gc"], dtype=np.float64)
    else:
        with open(handeye_path) as f:
            T_gc = np.array(yaml.safe_load(f)["T_gc"], dtype=np.float64)

    with open(intrinsics_path) as f:
        intrinsics = yaml.safe_load(f)

    return T_gc, intrinsics


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def scale_intrinsics(
    intrinsics: dict,
    orig_h: int,
    orig_w: int,
    target_h: int,
    target_w: int,
) -> dict:
    """Rescale fx, fy, cx, cy to match a resized image."""
    sx = target_w / orig_w
    sy = target_h / orig_h
    return dict(
        fx=intrinsics["fx"] * sx,
        fy=intrinsics["fy"] * sy,
        cx=intrinsics["cx"] * sx,
        cy=intrinsics["cy"] * sy,
    )


def depth_to_base_pointcloud(
    depth_mm: np.ndarray,
    intrinsics: dict,
    T_gc: np.ndarray,
    T_bg: np.ndarray | None = None,
) -> np.ndarray:
    """Project a depth image to 3D in the robot base frame.

    Full transform chain: P_base = T_bg @ T_gc @ P_camera
    If T_bg is None, output is in the gripper frame (T_gc only).

    Args:
        depth_mm:   (H, W) float32 in mm. 0 = invalid pixel.
        intrinsics: dict with fx, fy, cx, cy already scaled to depth_mm resolution.
        T_gc:       (4, 4) camera-to-gripper (hand-eye calibration, static).
        T_bg:       (4, 4) gripper-to-base (robot FK pose at capture time). None → gripper frame.

    Returns:
        (H, W, 3) float32 in base frame mm. NaN where depth == 0.
    """
    H, W = depth_mm.shape
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    d = depth_mm.astype(np.float64)
    valid = d > 0

    X_c = np.where(valid, (u - cx) * d / fx, 0.0)
    Y_c = np.where(valid, (v - cy) * d / fy, 0.0)
    Z_c = np.where(valid, d, 0.0)

    # (H, W, 4) homogeneous
    ones = np.ones((H, W), dtype=np.float64)
    pc_cam = np.stack([X_c, Y_c, Z_c, ones], axis=-1)

    T = T_gc if T_bg is None else T_bg @ T_gc
    pc_base = (pc_cam @ T.T)[..., :3].astype(np.float32)
    pc_base[~valid] = np.nan

    return pc_base


def compute_surface_normals(pc_base: np.ndarray) -> np.ndarray:
    """Estimate per-pixel surface normals from a 3D point cloud.

    Uses central differences on the XY neighbors to form two tangent vectors,
    then takes their cross product.

    Args:
        pc_base: (H, W, 3) float32 point cloud. NaN where invalid.

    Returns:
        (H, W, 3) float32 unit normals. NaN where the point or its neighbors are invalid.
    """
    # Central differences (border pixels get zero → will yield NaN normal anyway)
    dPdu = np.zeros_like(pc_base)
    dPdv = np.zeros_like(pc_base)
    dPdu[:, 1:-1] = pc_base[:, 2:] - pc_base[:, :-2]
    dPdv[1:-1, :] = pc_base[2:, :] - pc_base[:-2, :]

    normals = np.cross(dPdu, dPdv)  # (H, W, 3)

    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = np.where(norms > 1e-8, normals / norms, np.nan)

    return normals.astype(np.float32)


def _weighted_area_mean(
    img: np.ndarray,
    valid: np.ndarray,
    feat_H: int,
    feat_W: int,
) -> np.ndarray:
    """Pool (H, W, C) to (feat_H, feat_W, C) as a valid-pixel-weighted area mean.

    Uses cv2.INTER_AREA for efficiency.
    Returns NaN for patches with no valid pixels.
    """
    is_2d = img.ndim == 2
    if is_2d:
        img = img[..., None]
    C = img.shape[2]
    v_f = valid.astype(np.float32)

    # mean(valid) per patch — used as denominator
    cnt = cv2.resize(v_f, (feat_W, feat_H), interpolation=cv2.INTER_AREA)

    out = np.full((feat_H, feat_W, C), np.nan, dtype=np.float32)
    for c in range(C):
        ch = np.where(valid, img[..., c].astype(np.float32), 0.0)
        ch_mean = cv2.resize(ch, (feat_W, feat_H), interpolation=cv2.INTER_AREA)
        # mean(valid * ch) / mean(valid) = valid-weighted mean of ch
        with np.errstate(invalid="ignore", divide="ignore"):
            out[..., c] = np.where(cnt > 1e-6, ch_mean / cnt, np.nan)

    return out[..., 0] if is_2d else out


def compute_patch_geo_features(
    pc_base: np.ndarray,
    depth_mm: np.ndarray,
    feat_H: int,
    feat_W: int,
) -> np.ndarray:
    """Aggregate depth geometry into a DINO-compatible patch grid.

    Args:
        pc_base:  (H, W, 3) float32 3D points in robot base frame (mm). NaN = invalid.
        depth_mm: (H, W) float32 depth in mm (0 = invalid).
        feat_H:   number of patch rows (= DINO grid height, possibly × NAF scale).
        feat_W:   number of patch columns.

    Returns:
        (feat_H, feat_W, 7) float32:
            [0:3]  XYZ centroid in robot base frame (mm)
            [3:6]  mean surface normal in robot base frame (unit vector)
            [6]    depth standard deviation within patch (mm)
        NaN where a patch contains no valid depth pixels.
    """
    normals = compute_surface_normals(pc_base)  # (H, W, 3)

    valid_xyz = ~np.isnan(pc_base[..., 0])                    # (H, W)
    valid_nrm = valid_xyz & ~np.isnan(normals[..., 0])        # (H, W)
    valid_d   = depth_mm > 0                                   # (H, W)

    # --- XYZ centroid ---
    xyz_patches = _weighted_area_mean(pc_base, valid_xyz, feat_H, feat_W)  # (fH, fW, 3)

    # --- Surface normal (re-normalize after averaging) ---
    nrm_patches = _weighted_area_mean(normals, valid_nrm, feat_H, feat_W)  # (fH, fW, 3)
    nrm_norms = np.linalg.norm(nrm_patches, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        nrm_patches = np.where(nrm_norms > 1e-8, nrm_patches / nrm_norms, np.nan)

    # --- Depth std via E[x²] - E[x]² (vectorized, no loops) ---
    d_f = np.where(valid_d, depth_mm.astype(np.float32), 0.0)
    cnt_d    = cv2.resize(valid_d.astype(np.float32), (feat_W, feat_H), interpolation=cv2.INTER_AREA)
    mean_d   = cv2.resize(d_f,      (feat_W, feat_H), interpolation=cv2.INTER_AREA)
    mean_d2  = cv2.resize(d_f ** 2, (feat_W, feat_H), interpolation=cv2.INTER_AREA)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_d_w  = np.where(cnt_d > 1e-6, mean_d  / cnt_d, np.nan)
        mean_d2_w = np.where(cnt_d > 1e-6, mean_d2 / cnt_d, np.nan)
    std_d = np.sqrt(np.maximum(0.0, mean_d2_w - mean_d_w ** 2))  # (fH, fW)

    return np.concatenate([
        xyz_patches,       # (fH, fW, 3)
        nrm_patches,       # (fH, fW, 3)
        std_d[..., None],  # (fH, fW, 1)
    ], axis=-1).astype(np.float32)  # (fH, fW, 7)
