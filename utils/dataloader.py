import json
import yaml
import torch
import numpy as np
from typing import Any
from PIL import Image
from pathlib import Path
from pycocotools import mask as mask_utils
from torchvision.transforms import v2


def load_depth_paths(dataset_dir: str, annotation_file: str) -> list[str | None]:
    """Return depth image paths matching each COCO image (None if file not found).

    Convention: depth file = image filename with 'image_' replaced by 'depth_'.
    Example: 'image_00.png' → 'depth_00.png'
    """
    with open(annotation_file) as f:
        coco_data = json.load(f)
    dataset_path = Path(dataset_dir)
    paths = []
    for img_info in coco_data["images"]:
        fname = str(img_info["file_name"])
        depth_fname = fname.replace("image_", "depth_")
        depth_path = dataset_path / depth_fname
        paths.append(str(depth_path) if depth_path.exists() else None)
    return paths


def load_pose_paths(dataset_dir: str, annotation_file: str) -> list[str | None]:
    """Return robot pose file paths matching each COCO image (None if file not found).

    Convention: pose file = image filename with 'image_' → 'pose_' and '.png' → '.txt'.
    Example: 'image_00.png' → 'pose_00.txt'
    """
    with open(annotation_file) as f:
        coco_data = json.load(f)
    dataset_path = Path(dataset_dir)
    paths = []
    for img_info in coco_data["images"]:
        fname = str(img_info["file_name"])
        pose_fname = fname.replace("image_", "pose_").replace(".png", ".txt")
        pose_path = dataset_path / pose_fname
        paths.append(str(pose_path) if pose_path.exists() else None)
    return paths


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(stream=f)
    return config


def load_coco_images(
    dataset_dir: str,
    annotation_file: str,
    img_size: int,
    class_names: list[str] | None = None,
) -> tuple:
    """
    Load images and label maps from a COCO JSON annotation file.

    Args:
        class_names: if provided, build an integer label map per image instead of a binary mask.
            Label 0 = background, label i+1 = class_names[i].
            Annotations are applied in order so later classes override earlier ones
            (e.g. ["base", "welded"] carves welded regions out of base automatically).
    """
    with open(annotation_file, 'r') as f:
        coco_data: dict[str, object] = json.load(f)

    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    dataset_path: Path = Path(dataset_dir)

    image_to_anns: dict[int, list[dict[str, object]]] = {}
    for ann in coco_data.get('annotations', []):  # type: ignore[union-attr]
        image_to_anns.setdefault(ann['image_id'], []).append(ann)  # type: ignore[index]

    cat_name_to_id: dict[str, int] = {}
    if class_names is not None:
        cat_name_to_id = {cat['name']: cat['id'] for cat in coco_data.get('categories', [])}

    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    img_info: dict[str, object]
    for img_info in coco_data['images']:
        image: Image.Image = Image.open(dataset_path / str(img_info['file_name'])).convert('RGB')
        orig_h: int = int(img_info.get('height', image.height))
        orig_w: int = int(img_info.get('width', image.width))

        if class_names is not None:
            # Multi-class label map: apply classes in order so later labels override earlier ones
            label_map: np.ndarray = np.zeros((orig_h, orig_w), dtype=np.uint8)
            for j, name in enumerate(class_names):
                cat_id = cat_name_to_id.get(name)
                if cat_id is None:
                    continue
                for ann in image_to_anns.get(int(img_info['id']), []):
                    if ann.get('category_id') != cat_id:
                        continue
                    seg = ann.get('segmentation')
                    if isinstance(seg, list):
                        rle = mask_utils.merge(mask_utils.frPyObjects(seg, orig_h, orig_w))
                    elif isinstance(seg, dict):
                        rle = seg
                    else:
                        continue
                    label_map[mask_utils.decode(rle).astype(bool)] = j + 1
            mask_arr = label_map
        else:
            # Legacy binary mask: merge all annotations
            combined_mask: np.ndarray = np.zeros((orig_h, orig_w), dtype=np.uint8)
            ann: dict[str, object]
            for ann in image_to_anns.get(int(img_info['id']), []):
                segmentation: list[object] | dict[str, object] | None = ann.get('segmentation')
                if isinstance(segmentation, list):
                    rles: object = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
                    rle: object = mask_utils.merge(rles)
                elif isinstance(segmentation, dict):
                    rle = segmentation
                else:
                    continue
                combined_mask = np.maximum(combined_mask, mask_utils.decode(rle))
            mask_arr = combined_mask

        images.append(transform(image))
        mask_pil: Image.Image = Image.fromarray(mask_arr).resize(
            (img_size, img_size), Image.Resampling.NEAREST
        )
        masks.append(torch.from_numpy(np.array(mask_pil)).long())

    return torch.stack(images), torch.stack(masks)
