#!/usr/bin/env python3
import torch
import argparse
import numpy as np
from pathlib import Path

from utils.dataloader import load_config, load_coco_images
from utils.prototypes import build_prototypes, save_prototypes


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

    print(f"Loading support images from '{args.dataset_dir}' ({annotation})")
    print(f"Classes: {class_names}")
    sup_images, sup_masks = load_coco_images(
        args.dataset_dir, annotation, img_size, class_names=class_names
    )
    N = len(sup_images)
    print(f"Found {N} support image(s)")

    support_set = []
    for i in range(N):
        img_t   = sup_images[i].unsqueeze(0).to(device)
        mask_np = sup_masks[i].numpy().astype(np.uint8)
        support_set.append((img_t, mask_np))

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
    )
    save_prototypes(protos, config["proto_path"])
    shapes = {name: protos[name].shape for name in [*class_names, "background"]}
    print(f"Saved to '{config['proto_path']}' — " + "  ".join(f"{n}: {s}" for n, s in shapes.items()))


if __name__ == "__main__":
    main()
