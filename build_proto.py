#!/usr/bin/env python3
import torch
import argparse
import numpy as np
from pathlib import Path

from utils.dataloader import load_config, load_coco_images, load_depth_paths, load_pose_paths
from utils.prototypes import build_prototypes, save_prototypes
from utils.depth import (
    load_calibration, load_depth_mm, load_pose,
    scale_intrinsics, depth_to_base_pointcloud, compute_patch_geo_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DINO few-shot prototypes from a COCO dataset")
    parser.add_argument("--config",      default="configs/config_vitb.yaml")
    parser.add_argument("--dataset_dir", default="dataset", help="Support images directory (result.json must be inside)")
    parser.add_argument("--k", type=int, default=5, help="Prototypes per class")
    args = parser.parse_args()

    annotation = str(Path(args.dataset_dir) / "result.json")
    class_names = ["base", "welded"]

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = config["dino_size"]
    n_layers = config["dino_num_layers"]
    print(f"Device: {device}  |  img_size: {img_size}  |  layers: {n_layers}")

    print("Loading DINOv3 encoder")
    dino_encoder = torch.hub.load("dinov3", model=config["dino_model"], source="local", weights=config["dino_weights"]).to(device)
    dino_encoder.eval()

    naf_scale = config.get("naf_scale", 1)
    naf_model = None
    if naf_scale > 1:
        print(f"Loading NAF upsampler (scale={naf_scale})")
        naf_model = torch.hub.load("NAF", "naf", source="local", pretrained=True, device=str(device))
        naf_model.eval()

    # --- Depth calibration ---
    handeye_path    = config.get("handeye_config",    "configs/handeye.yaml")
    intrinsics_path = config.get("intrinsics_config", "configs/intrinsics.yaml")
    geo_weights = (
        config.get("depth_alpha", 0.3),
        config.get("depth_beta",  0.3),
        config.get("depth_gamma", 0.1),
    )
    depth_scale = config.get("depth_scale", 1000.0)

    T_gc, intrinsics = load_calibration(handeye_path, intrinsics_path)
    print(f"Loaded hand-eye T_gc from '{handeye_path}'")

    # --- Support images ---
    print(f"Loading support images from '{args.dataset_dir}' ({annotation})")
    print(f"Classes: {class_names}")
    sup_images, sup_masks = load_coco_images(
        args.dataset_dir, annotation, img_size, class_names=class_names
    )
    depth_paths = load_depth_paths(args.dataset_dir, annotation)
    pose_paths  = load_pose_paths(args.dataset_dir, annotation)
    N = len(sup_images)
    print(f"Found {N} support image(s)")

    use_depth = any(p is not None for p in depth_paths)
    if use_depth:
        print(f"Depth maps found: {sum(p is not None for p in depth_paths)}/{N}  |  "
              f"geo_weights=(α={geo_weights[0]}, β={geo_weights[1]}, γ={geo_weights[2]})  |  "
              f"depth_scale={depth_scale} mm")
    else:
        print("No depth maps found — building RGB-only prototypes")

    # DINO patch grid size after NAF upsampling
    patch_size = 16  # ViT patch size
    feat_size = (img_size // patch_size) * naf_scale  # e.g. 768/16 * 4 = 192

    support_set = []
    for i in range(N):
        img_t   = sup_images[i].unsqueeze(0).to(device)
        mask_np = sup_masks[i].numpy().astype(np.uint8)

        geo_features = None
        if depth_paths[i] is not None:
            # Load depth at original resolution to get true orig_h, orig_w
            import cv2
            depth_raw = cv2.imread(depth_paths[i], cv2.IMREAD_ANYDEPTH)
            orig_h, orig_w = depth_raw.shape[:2]

            # Resize depth to img_size × img_size and scale intrinsics accordingly
            depth_resized = load_depth_mm(depth_paths[i], target_hw=(img_size, img_size))
            intr_scaled   = scale_intrinsics(intrinsics, orig_h, orig_w, img_size, img_size)

            # Per-image robot pose (gripper → base); fall back to gripper frame if missing
            T_bg = load_pose(pose_paths[i]) if (pose_paths[i] is not None) else None
            if T_bg is None:
                print(f"  Image {i}: no pose file — using gripper frame")

            pc_base = depth_to_base_pointcloud(depth_resized, intr_scaled, T_gc, T_bg)
            geo_features = compute_patch_geo_features(
                pc_base, depth_resized, feat_size, feat_size
            )

        support_set.append((img_t, mask_np) if geo_features is None else (img_t, mask_np, geo_features))

    print(f"Building prototypes (k={args.k}, naf_scale={naf_scale})")
    protos = build_prototypes(
        support_set=support_set,
        dino_encoder=dino_encoder,
        dino_num_layers=n_layers,
        device=device,
        k=args.k,
        class_names=class_names,
        naf_model=naf_model,
        naf_scale=naf_scale,
        geo_weights=geo_weights,
        depth_scale=depth_scale,
    )
    save_prototypes(protos, config["proto_path"])
    shapes = {name: protos[name].shape for name in [*class_names, "background"]}
    print(f"Saved to '{config['proto_path']}' — " + "  ".join(f"{n}: {s}" for n, s in shapes.items()))


if __name__ == "__main__":
    main()
