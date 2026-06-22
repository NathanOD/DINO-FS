# DINO Few-Shot

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
python build_proto.py --dataset_dir test3 --config configs/config_vitl.yaml --k 4
```

Inference

```shell
python inference.py --config configs/config_vitl.yaml --query test2/dae7fdc0-image_04.png --intrinsics test/intrinsics.json
python inference.py --config configs/config_vitl.yaml --query test2 --intrinsics test/intrinsics.json --compute_metrics
python inference.py --config configs/config_vitl.yaml --query test3/79225dc1-image_00.png --intrinsics test/intrinsics.json --compute_metrics
```
