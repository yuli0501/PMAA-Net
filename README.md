# PMAA-Net
this is a demo
# Polymorphic multi-head attention aggregation network for skin lesion segmentation
## News
2024.7.2: The PMAA-Net model has been optimised. The paper will be updated later.

## Requirements
- PyTorch 1.x or 0.41

## Installation
1. Create an anaconda environment.
```sh
conda create -n=<env_name> python=3.9 anaconda
conda activate <env_name>
```
2. Install PyTorch.
```sh
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
3. Install pip packages.
```sh
pip install -r requirements.txt
```

## Training on [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018) dataset
1. Download dataset from [here]( https://challenge.isic-archive.com/data/#2018) to inputs/ and unzip. The file structure is the following:
```
inputs
└── ISIC2018
    ├── images
                ├──ISIC_00000001.png
                ├──ISIC_00000002.png
                ├──ISIC_0000000n.png
          ├── masks
                ├──ISIC_00000001.png
                ├──ISIC_00000002.png
                ├──ISIC_0000000n.png
    ...
```
```
2. Train the model.
```sh
V2.py
```
3. Evaluate.
```sh
V2.py
```

