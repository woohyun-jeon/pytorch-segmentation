# Semantic Segmentation in PyTorch

This repository contains the implementation of semantic segmentation models in PyTorch.

## Prerequisites
Install the required dependencies with the following command:
```
pip install -r requirements.txt
```


## Usage
1) Install the required dependencies with the following command:
```
pip install -r requirements.txt
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
- [ ] AttentionUNet
- [x] SegNet
- [x] HRNet
- [ ] DeepLabv2
- [ ] DeepLabv3
- [ ] DeepLabv3+
- [ ] PSPNet
- [ ] PAN
