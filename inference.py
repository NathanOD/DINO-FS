#!/usr/bin/env python3
import os
import cv2
import json
import time
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms import v2
from pycocotools import mask as coco_mask

from utils.dataloader import load_config
from utils.prototypes import load_prototypes, segment
from utils.visualize import overlay_label_map, compute_mean_iou
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


def load_coco_label_map(
    result_json_path: str,
    image_filename: str,
    h: int,
    w: int,
    class_names: list[str],
) -> np.ndarray:
    """Load a COCO annotation file and return an integer label map (H, W) uint8."""
    with open(result_json_path) as f:
        data = json.load(f)
    images = {img["id"]: img["file_name"] for img in data.get("images", [])}
    image_id = next(
        (iid for iid, fname in images.items() if fname == image_filename),
        None,
    )
    if image_id is None:
        raise ValueError(f"Image '{image_filename}' not found in '{result_json_path}'")
    cat_name_to_id = {cat["name"]: cat["id"] for cat in data.get("categories", [])}
    annotations = [a for a in data.get("annotations", []) if a["image_id"] == image_id]

    label_map = np.zeros((h, w), dtype=np.uint8)
    for j, name in enumerate(class_names):
        cat_id = cat_name_to_id.get(name)
        for ann in annotations:
            if ann.get("category_id") != cat_id:
                continue
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                rle = seg
            else:
                rle = coco_mask.merge(coco_mask.frPyObjects(seg, h, w))
            label_map[coco_mask.decode(rle).astype(bool)] = j + 1
    return label_map


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
) -> tuple[np.ndarray, np.ndarray]:
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

    pred_mask, _ = segment(
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
    return pred_mask, query_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot segmentation inference with saved prototypes")
    parser.add_argument("--query",      required=True, help="Query image path or directory (directory only with --compute_metrics)")
    parser.add_argument("--config",     default="configs/config_vitb.yaml")
    parser.add_argument("--compute_metrics", action="store_true", help="Compute IoU from result.json (COCO format) in the query image folder")
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

    if args.compute_metrics and os.path.isdir(args.query):
        result_json = os.path.join(args.query, "result.json")
        with open(result_json) as f:
            coco_data = json.load(f)
        images = coco_data.get("images", [])
        if not images:
            raise ValueError(f"No images listed in '{result_json}'")

        all_ious: list[dict] = []
        for img_info in images:
            query_path = os.path.join(args.query, img_info["file_name"])
            print(f"Processing '{img_info['file_name']}' …")
            t0 = time.time()
            pred_label_map, query_rgb = run_segment(query_path, **seg_kwargs)
            elapsed = time.time() - t0
            H, W = query_rgb.shape[:2]
            gt_label_map = load_coco_label_map(result_json, img_info["file_name"], H, W, class_names)
            if gt_label_map.shape != pred_label_map.shape:
                gt_label_map = np.array(Image.fromarray(gt_label_map).resize(
                    (pred_label_map.shape[1], pred_label_map.shape[0]), Image.NEAREST
                ))
            ious = compute_mean_iou(pred_label_map, gt_label_map, class_names)
            all_ious.append(ious)
            iou_str = "  ".join(f"{n}: {v:.4f}" for n, v in ious.items())
            print(f"  Inference: {elapsed:.2f}s  |  {iou_str}")
        mean_ious = {n: float(np.mean([d[n] for d in all_ious])) for n in class_names}
        print(f"\nMean IoU over {len(all_ious)} images: " + "  ".join(f"{n}: {v:.4f}" for n, v in mean_ious.items()))

    else:
        print(f"Segmenting '{args.query}'")
        t0 = time.time()
        pred_label_map, query_rgb = run_segment(args.query, **seg_kwargs)
        print(f"Inference time: {time.time() - t0:.2f}s")

        H_orig, W_orig = query_rgb.shape[:2]
        overlay_label_map(query_rgb, pred_label_map, class_names, save_path="result.png")

        if args.compute_metrics:
            result_json = os.path.join(os.path.dirname(os.path.abspath(args.query)), "result.json")
            gt_label_map = load_coco_label_map(result_json, os.path.basename(args.query), H_orig, W_orig, class_names)
            if gt_label_map.shape != pred_label_map.shape:
                gt_label_map = np.array(Image.fromarray(gt_label_map).resize(
                    (pred_label_map.shape[1], pred_label_map.shape[0]), Image.NEAREST
                ))
            ious = compute_mean_iou(pred_label_map, gt_label_map, class_names)
            print("IoU vs ground truth: " + "  ".join(f"{n}: {v:.4f}" for n, v in ious.items()))


if __name__ == "__main__":
    main()
