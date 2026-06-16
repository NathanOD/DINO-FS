#!/usr/bin/env python3
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

from utils.dataloader import load_config, load_coco_images
from utils.prototypes import build_prototypes, save_prototypes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DINO few-shot prototypes from a COCO dataset")
    parser.add_argument("--config",      default="configs/config_vitb.yaml")
    parser.add_argument("--dataset_dir", default="dataset",           help="Support images directory (result.json must be inside)")
    parser.add_argument("--k", type=int, default=5, help="Prototypes per class")
    args = parser.parse_args()

    annotation = str(Path(args.dataset_dir) / "result.json")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = config["dino_size"]
    n_layers = config["dino_num_layers"]
    print(f"Device: {device}  |  img_size: {img_size}  |  layers: {n_layers}")

    print("Loading DINOv3 encoder")
    dino_encoder = torch.hub.load("dinov3", config["dino_model"], source="local").to(device)
    dino_encoder.eval()

    print(f"Loading support images from '{args.dataset_dir}' ({annotation})")
    sup_images, sup_masks = load_coco_images(args.dataset_dir, annotation, img_size)
    N = len(sup_images)
    print(f"Found {N} support image(s)")

    support_set = []
    for i in range(N):
        img_t   = sup_images[i].unsqueeze(0).to(device)
        mask_np = sup_masks[i].numpy().astype(np.uint8)
        support_set.append((img_t, mask_np))

    print(f"Building prototypes (k={args.k})")
    protos = build_prototypes(
        support_set=support_set,
        dino_encoder=dino_encoder,
        dino_num_layers=n_layers,
        device=device,
        k=args.k,
    )
    save_prototypes(protos, config["proto_path"])
    print(
        f"Saved to '{config["proto_path"]}' - "
        f"[piece: {protos['piece'].shape}  bg: {protos['background'].shape}]"
    )


if __name__ == "__main__":
    main()
