#!/usr/bin/env python3
import json
import os
import time
import argparse

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask

from utils.dataloader import load_config
from utils.prototypes import load_prototypes
from utils.depth import load_calibration
from inference import run_segment


def load_coco_label_map(
    result_json_path: str,
    image_filename: str,
    h: int,
    w: int,
    class_names: list[str],
) -> np.ndarray:
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
            rle = seg if isinstance(seg, dict) else coco_mask.merge(coco_mask.frPyObjects(seg, h, w))
            label_map[coco_mask.decode(rle).astype(bool)] = j + 1
    return label_map


def _class_ious(
    pred: np.ndarray,
    gt: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    """Per-class IoU. Returns 1.0 when both pred and GT are empty for a class."""
    result = {}
    for j, name in enumerate(class_names):
        label = j + 1
        p = pred == label
        g = gt == label
        intersection = int((p & g).sum())
        union = int((p | g).sum())
        result[name] = 1.0 if union == 0 else intersection / union
    return result


def _compute_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (pixel-level).

    Returns nan if GT has no positive pixels.
    """
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted).astype(np.float64)
    fp = np.cumsum(1 - labels_sorted).astype(np.float64)
    recall    = tp / n_pos
    precision = tp / (tp + fp)
    recall    = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapezoid(precision, recall))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute IoU / AP metrics over a test folder (requires result.json in COCO format)"
    )
    parser.add_argument("--test_dir", required=True, help="Folder containing images and result.json")
    parser.add_argument("--config",   default="configs/config_vitb.yaml")
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

    result_json = os.path.join(args.test_dir, "result.json")
    with open(result_json) as f:
        coco_data = json.load(f)
    images = coco_data.get("images", [])
    if not images:
        raise ValueError(f"No images listed in '{result_json}'")

    all_ious: list[dict] = []
    # Accumulate per-class (score, gt_label) pairs at patch resolution across all images
    score_accum: dict[str, list[np.ndarray]] = {n: [] for n in class_names}
    gt_accum:    dict[str, list[np.ndarray]] = {n: [] for n in class_names}

    for img_info in images:
        query_path = os.path.join(args.test_dir, img_info["file_name"])
        print(f"Processing '{img_info['file_name']}' …")
        t0 = time.time()
        pred_label_map, query_rgb, score_maps = run_segment(query_path, **seg_kwargs)
        elapsed = time.time() - t0

        H, W = query_rgb.shape[:2]
        gt_label_map = load_coco_label_map(result_json, img_info["file_name"], H, W, class_names)
        if gt_label_map.shape != pred_label_map.shape:
            gt_label_map = np.array(Image.fromarray(gt_label_map).resize(
                (pred_label_map.shape[1], pred_label_map.shape[0]), Image.NEAREST
            ))

        ious = _class_ious(pred_label_map, gt_label_map, class_names)
        all_ious.append(ious)
        iou_str = "  ".join(f"{n}: {v:.4f}" for n, v in ious.items())
        print(f"  Inference: {elapsed:.2f}s  |  IoU — {iou_str}")

        # Downsample GT to patch resolution and accumulate for AP
        for j, name in enumerate(class_names):
            sm = score_maps[name]           # (Hp, Wp) float32
            Hp, Wp = sm.shape
            gt_binary = (gt_label_map == j + 1).astype(np.uint8)
            gt_patch = np.array(
                Image.fromarray(gt_binary).resize((Wp, Hp), Image.NEAREST)
            ).ravel()
            score_accum[name].append(sm.ravel())
            gt_accum[name].append(gt_patch)

    # --- Per-image mean IoU ---
    mean_ious = {n: float(np.mean([d[n] for d in all_ious])) for n in class_names}
    print(f"\nMean IoU over {len(all_ious)} images: " +
          "  ".join(f"{n}: {v:.4f}" for n, v in mean_ious.items()))

    # --- Pixel-level AP per class and mAP ---
    aps: dict[str, float] = {}
    for name in class_names:
        scores_all = np.concatenate(score_accum[name])
        labels_all = np.concatenate(gt_accum[name]).astype(np.int32)
        aps[name] = _compute_ap(scores_all, labels_all)

    valid_aps = [v for v in aps.values() if not np.isnan(v)]
    map_score = float(np.mean(valid_aps)) if valid_aps else float("nan")

    ap_str = "  ".join(
        f"{n}: {'nan' if np.isnan(v) else f'{v:.4f}'}" for n, v in aps.items()
    )
    print(f"AP per class:  {ap_str}")
    print(f"mAP:           {map_score:.4f}" if not np.isnan(map_score) else "mAP: nan")


if __name__ == "__main__":
    main()
