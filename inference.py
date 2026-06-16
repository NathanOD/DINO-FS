#!/usr/bin/env python3
import json
import time
import torch
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from torchvision.transforms import v2

from utils.dataloader import load_config
from utils.prototypes import load_prototypes, segment
from utils.visualize import overlay_mask, compute_iou


def load_query_tensor(path: str, img_size: int, device: torch.device) -> torch.Tensor:
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def load_mask_np(path: str) -> np.ndarray:
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)


def mask_to_pointcloud(
    mask: np.ndarray,
    depth: np.ndarray,
    rgb: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    depth_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    z = depth[ys, xs].astype(np.float64) * depth_scale
    valid = z > 0
    xs, ys, z = xs[valid], ys[valid], z[valid]
    points = np.stack([
        (xs - cx) * z / fx,
        (ys - cy) * z / fy,
        z,
    ], axis=1)
    colors = rgb[ys, xs]
    return points, colors


def save_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(path, pcd, write_ascii=False)
    print(f"Saved point cloud to '{path}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot segmentation inference with saved prototypes")
    parser.add_argument("--query",      required=True, help="Query image path (must contain 'image' in filename)")
    parser.add_argument("--intrinsics", required=True, help="Camera intrinsics JSON (fx, fy, cx, cy[, depth_scale])")
    parser.add_argument("--config",     default="configs/config_vitb.yaml")
    parser.add_argument("--gt_mask",    default=None,  help="GT mask (optional, for IoU)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = config["dino_size"]
    n_layers = config["dino_num_layers"]
    print(f"Device: {device}  |  img_size: {img_size}  |  layers: {n_layers}")

    print("Loading DINOv3 encoder …")
    dino_encoder = torch.hub.load("dinov3", config["dino_model"], source="local").to(device)
    dino_encoder.eval()

    proto_path = config["proto_path"]
    print(f"Loading prototypes from '{proto_path}'")
    protos = load_prototypes(proto_path)
    print(f"piece: {protos['piece'].shape}  bg: {protos['background'].shape}")

    with open(args.intrinsics) as f:
        K = json.load(f)
    fx, fy = float(K["fx"]), float(K["fy"])
    cx, cy = float(K["cx"]), float(K["cy"])
    depth_scale = float(K.get("depth_scale", 1.0))

    print(f"Segmenting '{args.query}'")
    start_time = time.time()
    query_tensor = load_query_tensor(args.query, img_size, device)
    query_rgb    = np.array(Image.open(args.query).convert("RGB"))

    pred_mask, _ = segment(
        image_tensor=query_tensor,
        protos=protos,
        dino_encoder=dino_encoder,
        dino_num_layers=n_layers,
        device=device,
        tau=config["tau"],
        min_area=config["min_area"],
        morph_kernel=config["morph_kernel"],
    )

    end_time = time.time()

    H_orig, W_orig = query_rgb.shape[:2]
    if pred_mask.shape != (H_orig, W_orig):
        pred_mask = np.array(
            Image.fromarray(pred_mask).resize((W_orig, H_orig), Image.NEAREST)
        )

    print(f"Inference time: {end_time - start_time:.2f}s")
    overlay_mask(query_rgb, pred_mask, alpha=0.5, color=(0, 255, 0), save_path="result.png")

    depth_path = args.query.replace("image", "depth")
    depth_raw = np.array(Image.open(depth_path))
    points, colors = mask_to_pointcloud(
        pred_mask, depth_raw, query_rgb, fx, fy, cx, cy, depth_scale
    )
    save_ply("result.ply", points, colors)

    if args.gt_mask is not None:
        gt_np = load_mask_np(args.gt_mask)
        if gt_np.shape != pred_mask.shape:
            gt_np = np.array(
                Image.fromarray(gt_np).resize((pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST)
            )
        print(f"IoU vs ground truth: {compute_iou(pred_mask, gt_np):.4f}")


if __name__ == "__main__":
    main()
