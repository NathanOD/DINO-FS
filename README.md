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

```shell
git clone https://github.com/valeoai/NAF.git
pip install natten==0.21.6+torch2120cu132 -f https://whl.natten.org
````

## Few-Shot Segmentation

Prototype Generation

```shell
python build_proto.py --dataset_dir dataset2 --config configs/config_vitl.yaml
python build_proto.py --dataset_dir dataset2 --config configs/config_vithplus.yaml
```

Inference

```shell
python inference.py --config configs/config_vitl.yaml --query test/image_00.png
python inference.py --config configs/config_vithplus.yaml --query test/image_00.png
```
