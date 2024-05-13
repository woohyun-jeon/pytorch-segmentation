# Semantic Segmentation in PyTorch

This repository contains the implementation of semantic segmentation models in PyTorch.

## Prerequisites
* python >= 3.6
* torch >= 1.8.1
* torchvision >= 0.9.1


## Usage
1) Clone the repository and install the required dependencies with the following command:
```
$ git clone https://github.com/woohyun-jeon/pytorch-segmentation.git
$ cd pytorch-segmentation
$ pip install -r requirements.txt
```
2) Download [CityScapes](https://www.cityscapes-dataset.com/) into datasets directory

The directory structure should be as follows:
```
  datasets/
    CityScapes/      
      gtFine/
        train/
            aachen/
                *_gtFine_labelIds.png
                ...
            bochum/
                *_gtFine_labelIds.png
                ...
            ...
        val/
        test/
      leftImg8bit/
        train/
            aachen/
                *_leftImg8bit.png
                ...
            bochum/
                *_leftImg8bit.png
                ...
            ...
        val/
        test/
      
```

3) Run ```python train.py``` for training

## Supported Models
- [x] UNet
- [x] UNet++
- [x] UNet3+
- [x] AttentionUNet
- [x] Swin-UNet
- [x] SegNet
- [x] HRNet
- [x] DeepLabv3
- [x] DeepLabv3+
- [x] PSPNet
- [x] PAN
