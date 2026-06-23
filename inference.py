#!/usr/bin/env python3
import os
import json
import time
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision.transforms import v2
from pycocotools import mask as coco_mask

from utils.dataloader import load_config
from utils.prototypes import load_prototypes, segment
from utils.visualize import overlay_label_map, compute_mean_iou


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


def run_segment(
    query_path: str,
    img_size: int,
    device: torch.device,
    dino_encoder,
    protos: dict,
    n_layers: int,
    config: dict,
    naf_model=None,
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
        naf_model=naf_model,
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
    print(f"Classes: {class_names}  |  bg: {protos['background'].shape}")

    seg_kwargs = dict(
        img_size=img_size, device=device, dino_encoder=dino_encoder,
        protos=protos, n_layers=n_layers, config=config, naf_model=naf_model,
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
