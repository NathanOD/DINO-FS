#!/usr/bin/env python3
import time
import torch
import argparse
import numpy as np
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot segmentation inference with saved prototypes")
    parser.add_argument("--query", required=True, help="Query image path")
    parser.add_argument("--config", default="configs/config_vitb.yaml")
    parser.add_argument("--out_path", default="result_overlay.png")
    parser.add_argument("--gt_mask",  default=None, help="GT mask (optional, for IoU)")
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

    # Resize mask from dino_size back to original query resolution
    H_orig, W_orig = query_rgb.shape[:2]
    if pred_mask.shape != (H_orig, W_orig):
        pred_mask = np.array(
            Image.fromarray(pred_mask).resize((W_orig, H_orig), Image.NEAREST)
        )

    print(f"Inference time: {end_time-start_time:.2f}s")
    overlay_mask(query_rgb, pred_mask, alpha=0.5, color=(0, 255, 0), save_path=args.out_path)

    if args.gt_mask is not None:
        gt_np = load_mask_np(args.gt_mask)
        if gt_np.shape != pred_mask.shape:
            gt_np = np.array(
                Image.fromarray(gt_np).resize((pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST)
            )
        print(f"IoU vs ground truth: {compute_iou(pred_mask, gt_np):.4f}")


if __name__ == "__main__":
    main()
