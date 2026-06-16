import json
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from pycocotools import mask as mask_utils
from torchvision.transforms import v2


def load_config(config_path: str) -> dict[str, object]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(stream=f)
    return config


def load_coco_images(
    dataset_dir: str,
    annotation_file: str,
    img_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load images and masks from a COCO JSON annotation file."""
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

    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    image_names: list[str] = []

    img_info: dict[str, object]
    for img_info in coco_data['images']:
        image: Image.Image = Image.open(dataset_path / str(img_info['file_name'])).convert('RGB')
        orig_h: int = int(img_info.get('height', image.height))
        orig_w: int = int(img_info.get('width', image.width))

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

        images.append(transform(image))
        mask_pil: Image.Image = Image.fromarray(combined_mask).resize(
            (img_size, img_size), Image.Resampling.NEAREST
        )
        masks.append(torch.from_numpy(np.array(mask_pil)).long())
        #image_names.append(str(img_info['file_name']))

    return torch.stack(images), torch.stack(masks)#, image_names
