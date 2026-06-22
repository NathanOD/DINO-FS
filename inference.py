#!/usr/bin/env python3
import os
import json
import time
import torch
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from torchvision.transforms import v2
from pycocotools import mask as coco_mask

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


def load_coco_mask(result_json_path: str, image_filename: str, h: int, w: int) -> np.ndarray:
    with open(result_json_path) as f:
        data = json.load(f)
    images = {img["id"]: img["file_name"] for img in data.get("images", [])}
    image_id = next(
        (iid for iid, fname in images.items() if fname == image_filename),
        None,
    )
    if image_id is None:
        raise ValueError(f"Image '{image_filename}' not found in '{result_json_path}'")
    annotations = [a for a in data.get("annotations", []) if a["image_id"] == image_id]
    if not annotations:
        raise ValueError(f"No annotation for image '{image_filename}' in '{result_json_path}'")
    seg = annotations[0]["segmentation"]
    if isinstance(seg, dict):
        rle = coco_mask.frPyObjects(seg, h, w)
    else:
        rle = coco_mask.merge(coco_mask.frPyObjects(seg, h, w))
    return coco_mask.decode(rle).astype(np.uint8)


def run_segment(
    query_path: str,
    img_size: int,
    device: torch.device,
    dino_encoder,
    protos: dict,
    n_layers: int,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    query_tensor = load_query_tensor(query_path, img_size, device)
    query_rgb = np.array(Image.open(query_path).convert("RGB"))
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
    H, W = query_rgb.shape[:2]
    if pred_mask.shape != (H, W):
        pred_mask = np.array(Image.fromarray(pred_mask).resize((W, H), Image.NEAREST))
    return pred_mask, query_rgb


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
    parser.add_argument("--query",      required=True, help="Query image path or directory (directory only with --compute_metrics)")
    parser.add_argument("--intrinsics", default=None,  help="Camera intrinsics JSON (fx, fy, cx, cy, depth_scale)")
    parser.add_argument("--config",     default="configs/config_vitb.yaml")
    parser.add_argument("--compute_metrics", action="store_true", help="Compute IoU from result.json (COCO format) in the query image folder")
    parser.add_argument("--hand_eye",   default="test/hand_eye.json", help="Hand-eye calibration JSON (T_gc)")
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

    seg_kwargs = dict(
        img_size=img_size, device=device, dino_encoder=dino_encoder,
        protos=protos, n_layers=n_layers, config=config,
    )

    if args.compute_metrics and os.path.isdir(args.query):
        result_json = os.path.join(args.query, "result.json")
        with open(result_json) as f:
            coco_data = json.load(f)
        images = coco_data.get("images", [])
        if not images:
            raise ValueError(f"No images listed in '{result_json}'")

        ious = []
        for img_info in images:
            query_path = os.path.join(args.query, img_info["file_name"])
            print(f"Processing '{img_info['file_name']}' …")
            t0 = time.time()
            pred_mask, query_rgb = run_segment(query_path, **seg_kwargs)
            elapsed = time.time() - t0
            H, W = query_rgb.shape[:2]
            gt_np = load_coco_mask(result_json, img_info["file_name"], H, W)
            if gt_np.shape != pred_mask.shape:
                gt_np = np.array(Image.fromarray(gt_np).resize((pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST))
            iou = compute_iou(pred_mask, gt_np)
            ious.append(iou)
            print(f"  Inference: {elapsed:.2f}s  |  IoU: {iou:.4f}")
        print(f"\nMean IoU over {len(ious)} images: {np.mean(ious):.4f}")

    else:
        if not args.compute_metrics and args.intrinsics is None:
            raise ValueError("--intrinsics is required when not using --compute_metrics")

        print(f"Segmenting '{args.query}'")
        t0 = time.time()
        pred_mask, query_rgb = run_segment(args.query, **seg_kwargs)
        print(f"Inference time: {time.time() - t0:.2f}s")

        H_orig, W_orig = query_rgb.shape[:2]
        overlay_mask(query_rgb, pred_mask, alpha=0.5, color=(0, 255, 0), save_path="result.png")

        if not args.compute_metrics:
            with open(args.intrinsics) as f:
                K = json.load(f)
            fx, fy = float(K["fx"]), float(K["fy"])
            cx, cy = float(K["cx"]), float(K["cy"])
            depth_scale = float(K.get("depth_scale", 1.0))

            with open(args.hand_eye) as f:
                T_gc = np.array(json.load(f)["T_gc"], dtype=np.float64)

            pose_path = args.query.replace("image", "pose").replace(".png", ".txt")
            with open(pose_path) as f:
                raw = f.read()
            rows = [r.strip() for r in raw.strip().strip("[];").split(";") if r.strip()]
            T_robot = np.array([[float(v) for v in r.split(",")] for r in rows], dtype=np.float64)
            T_total = T_robot @ T_gc

            depth_path = args.query.replace("image", "depth")
            depth_raw = np.array(Image.open(depth_path))
            points, colors = mask_to_pointcloud(
                pred_mask, depth_raw, query_rgb, fx, fy, cx, cy, depth_scale
            )
            points = (T_total[:3, :3] @ points.T).T + T_total[:3, 3]
            save_ply("result.ply", points, colors)

        else:
            result_json = os.path.join(os.path.dirname(os.path.abspath(args.query)), "result.json")
            gt_np = load_coco_mask(result_json, os.path.basename(args.query), H_orig, W_orig)
            if gt_np.shape != pred_mask.shape:
                gt_np = np.array(Image.fromarray(gt_np).resize((pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST))
            print(f"IoU vs ground truth: {compute_iou(pred_mask, gt_np):.4f}")


if __name__ == "__main__":
    main()
