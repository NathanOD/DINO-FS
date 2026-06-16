# DINO-SAM-FS

## Installation

```shell
git clone https://github.com/facebookresearch/dinov3.git
````

```shell
conda create -n dinofs python=3.12
conda activate dinofs
pip install -r requirements.txt
```

## Few-Shot Segmentation

Prototype Generation

```shell
python build_proto.py --dataset_dir dataset --config configs/config_vitl.yaml
```

Inference

```shell
python inference.py --config configs/config_vitl.yaml --query queries/image_0019.png --out_path result.png
```
