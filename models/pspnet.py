import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.resnet import resnet50, resnet101, resnet152


__all__ = ['psp_resnet50', 'psp_resnet101', 'psp_resnet152']


class PSPModule(nn.Module):
    def __init__(self, in_dims, inner_dims, sizes=[1,2,3,6]):
        super(PSPModule, self).__init__()
        self.features = []
        for size in sizes:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(size,size)),
                    nn.Conv2d(in_dims, inner_dims, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=inner_dims),
                    nn.ReLU(inplace=True)
                )
            )
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        out = []
        out.append(x)
        for feature in self.features:
            out.append(F.interpolate(feature(x), size=x.shape[-2:], mode='bilinear', align_corners=True))

        out_psp = torch.cat(out, dim=1)

        return out_psp


class PSPNet(nn.Module):
    def __init__(self, in_channels, num_classes, model='resnet50', sizes=[1,2,3,6]):
        super(PSPNet, self).__init__()
        feature_dims = 2048
        inner_dims = int(feature_dims//len(sizes))

        if model == 'resnet50':
            self.backbone = resnet50(in_channels=in_channels)
        elif model == 'resnet101':
            self.backbone = resnet101(in_channels=in_channels)
        elif model == 'resnet152':
            self.backbone = resnet152(in_channels=in_channels)

        self.pspmodule = PSPModule(in_dims=feature_dims, inner_dims=inner_dims, sizes=sizes)
        self.classifier = nn.Sequential(
            nn.Conv2d(feature_dims*2, inner_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=inner_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dims, num_classes, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        out, _ = self.backbone(x)
        out = self.pspmodule(out)
        out = self.classifier(out)

        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)

        return out


def psp_resnet50(**kwargs):
    model = PSPNet(model='resnet50', **kwargs)

    return model


def psp_resnet101(**kwargs):
    model = PSPNet(model='resnet101', **kwargs)

    return model


def psp_resnet152(**kwargs):
    model = PSPNet(model='resnet152', **kwargs)

    return model


if __name__ == '__main__':
    img_size = 512
    model = psp_resnet101(in_channels=3, num_classes=1000)

    input = torch.randn(4, 3, img_size, img_size)
    output = model(input)

    print(output.shape)