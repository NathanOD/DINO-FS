#!/usr/bin/env python3
import cv2
import time
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms import v2

from utils.dataloader import load_config
from utils.prototypes import load_prototypes, segment
from utils.visualize import overlay_label_map
from utils.depth import (
    load_calibration, load_depth_mm, load_pose,
    scale_intrinsics, depth_to_base_pointcloud, compute_patch_geo_features,
)


def load_query_tensor(path: str, img_size: int, device: torch.device) -> torch.Tensor:
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)



def _derive_depth_path(image_path: str) -> str | None:
    """Return depth image path from RGB image path (None if not found).

    Convention: 'image_XX.png' → 'depth_XX.png' in the same directory.
    """
    p = Path(image_path)
    depth_name = p.name.replace("image_", "depth_")
    depth_path = p.parent / depth_name
    return str(depth_path) if depth_path.exists() else None


def _derive_pose_path(image_path: str) -> str | None:
    """Return robot pose file path from RGB image path (None if not found).

    Convention: 'image_XX.png' → 'pose_XX.txt' in the same directory.
    """
    p = Path(image_path)
    pose_name = p.name.replace("image_", "pose_").replace(".png", ".txt")
    pose_path = p.parent / pose_name
    return str(pose_path) if pose_path.exists() else None


def load_query_geo_features(
    query_path: str,
    img_size: int,
    T_gc: np.ndarray,
    intrinsics: dict,
    feat_size: int,
) -> np.ndarray | None:
    """Load depth + pose for the query image and return (feat_size, feat_size, 7) geo features.

    Returns None if no depth image is found alongside the query.
    """
    depth_path = _derive_depth_path(query_path)
    if depth_path is None:
        return None

    depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    orig_h, orig_w = depth_raw.shape[:2]
    depth_resized = load_depth_mm(depth_path, target_hw=(img_size, img_size))
    intr_scaled   = scale_intrinsics(intrinsics, orig_h, orig_w, img_size, img_size)

    pose_path = _derive_pose_path(query_path)
    T_bg = load_pose(pose_path) if pose_path is not None else None

    pc_base = depth_to_base_pointcloud(depth_resized, intr_scaled, T_gc, T_bg)
    return compute_patch_geo_features(pc_base, depth_resized, feat_size, feat_size)


def run_segment(
    query_path: str,
    img_size: int,
    device: torch.device,
    dino_encoder,
    protos: dict,
    n_layers: int,
    config: dict,
    naf_model=None,
    T_gc: np.ndarray | None = None,
    intrinsics: dict | None = None,
    feat_size: int = 48,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Returns (pred_label_map, query_rgb, score_maps).

    score_maps: {class_name: (Hp, Wp) float32} — sim(class)−sim(bg) at patch resolution.
    """
    query_tensor = load_query_tensor(query_path, img_size, device)
    query_rgb = np.array(Image.open(query_path).convert("RGB"))

    geo_features = None
    if protos.get("use_depth") and T_gc is not None and intrinsics is not None:
        geo_features = load_query_geo_features(query_path, img_size, T_gc, intrinsics, feat_size)
        if geo_features is None:
            print(f"  Warning: prototypes were built with depth but no depth found for '{query_path}'")

    geo_weights = (
        config.get("depth_alpha", 0.3),
        config.get("depth_beta",  0.3),
        config.get("depth_gamma", 0.1),
    )
    depth_scale = config.get("depth_scale", 1000.0)

    pred_mask, score_maps = segment(
        image_tensor=query_tensor,
        protos=protos,
        dino_encoder=dino_encoder,
        dino_num_layers=n_layers,
        device=device,
        tau=config["tau"],
        min_area=config["min_area"],
        morph_kernel=config["morph_kernel"],
        naf_model=naf_model,
        geo_features=geo_features,
        geo_weights=geo_weights,
        depth_scale=depth_scale,
    )
    H, W = query_rgb.shape[:2]
    if pred_mask.shape != (H, W):
        pred_mask = np.array(Image.fromarray(pred_mask).resize((W, H), Image.NEAREST))
    return pred_mask, query_rgb, score_maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot segmentation inference with saved prototypes")
    parser.add_argument("--query",  required=True, help="Query image path")
    parser.add_argument("--config", default="configs/config_vitb.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = config["dino_size"]
    n_layers = config["dino_num_layers"]
    print(f"Device: {device}  |  img_size: {img_size}  |  layers: {n_layers}")

    print("Loading DINOv3 encoder …")
    dino_encoder = torch.hub.load("dinov3", config["dino_model"], source="local").to(device)
    dino_encoder.eval()

    naf_model = None
    naf_scale = config.get("naf_scale", 1)
    if naf_scale > 1:
        print(f"Loading NAF upsampler (scale={naf_scale}) …")
        naf_model = torch.hub.load("NAF", "naf", source="local", pretrained=True, device=str(device))
        naf_model.eval()

    proto_path = config["proto_path"]
    print(f"Loading prototypes from '{proto_path}'")
    protos = load_prototypes(proto_path)
    class_names = protos.get("class_names", ["piece"])
    print(f"Classes: {class_names}  |  bg: {protos['background'].shape}  |  "
          f"depth: {protos.get('use_depth', False)}")

    # --- Depth calibration (only needed if prototypes were built with depth) ---
    T_gc, intrinsics = None, None
    if protos.get("use_depth"):
        handeye_path    = config.get("handeye_config",    "configs/handeye.yaml")
        intrinsics_path = config.get("intrinsics_config", "configs/intrinsics.yaml")
        T_gc, intrinsics = load_calibration(handeye_path, intrinsics_path)
        print(f"Loaded hand-eye T_gc from '{handeye_path}'")

    patch_size = 16
    feat_size = (img_size // patch_size) * naf_scale

    seg_kwargs = dict(
        img_size=img_size, device=device, dino_encoder=dino_encoder,
        protos=protos, n_layers=n_layers, config=config, naf_model=naf_model,
        T_gc=T_gc, intrinsics=intrinsics, feat_size=feat_size,
    )

    print(f"Segmenting '{args.query}'")
    t0 = time.time()
    pred_label_map, query_rgb, _ = run_segment(args.query, **seg_kwargs)
    print(f"Inference time: {time.time() - t0:.2f}s")

    overlay_label_map(query_rgb, pred_label_map, class_names, save_path="result.png")


if __name__ == "__main__":
    main()
