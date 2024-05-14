import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.resnet import resnet50, resnet101, resnet152


__all__ = ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_resnet152']


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, atrous_rates=[6,12,18]):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True)
        )
        self.conv3x3a = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=1, padding=atrous_rates[0], dilation=atrous_rates[0], bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True)
        )
        self.conv3x3b = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=1, padding=atrous_rates[1], dilation=atrous_rates[1], bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True)
        )
        self.conv3x3c = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=1, padding=atrous_rates[2], dilation=atrous_rates[2], bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True)
        )
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_dims*5, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out1 = self.conv1x1(x)
        out2 = self.conv3x3a(x)
        out3 = self.conv3x3b(x)
        out4 = self.conv3x3c(x)
        out5 = self.pooling(x)
        out5 = F.interpolate(out5, size=x.shape[-2:], mode='bilinear', align_corners=True)

        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.conv_cat(out)

        return out

class DeepLabv3(nn.Module):
    def __init__(self, in_channels, num_classes, model='resnet50', output_stride=16, atrous_rates=[6,12,18]):
        super(DeepLabv3, self).__init__()
        low_level_dims = 256
        feature_dims = 2048
        if model == 'resnet50':
            self.backbone = resnet50(in_channels=in_channels, output_stride=output_stride)
        elif model == 'resnet101':
            self.backbone = resnet101(in_channels=in_channels, output_stride=output_stride)
        elif model == 'resnet152':
            self.backbone = resnet152(in_channels=in_channels, output_stride=output_stride)

        self.aspp = ASPP(in_dims=feature_dims, out_dims=low_level_dims, atrous_rates=atrous_rates)
        self.conv = nn.Conv2d(low_level_dims, num_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out, _ = self.backbone(x)
        out = self.aspp(out)
        out = self.conv(out)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)

        return out


def deeplabv3_resnet50(**kwargs):
    model = DeepLabv3(model='resnet50', **kwargs)

    return model


def deeplabv3_resnet101(**kwargs):
    model = DeepLabv3(model='resnet101', **kwargs)

    return model


def deeplabv3_resnet152(**kwargs):
    model = DeepLabv3(model='resnet152', **kwargs)

    return model


if __name__ == '__main__':
    img_size = 512
    model = deeplabv3_resnet50(in_channels=3, num_classes=1000)

    input = torch.randn(4, 3, img_size, img_size)
    output = model(input)

    print(output.shape)
