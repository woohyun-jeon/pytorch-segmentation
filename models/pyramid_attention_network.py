import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.resnet import resnet50, resnet101, resnet152


__all__ = ['pan_resnet50', 'pan_resnet101', 'pan_resnet152']


class FeaturePyramidAttention(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(FeaturePyramidAttention, self).__init__()
        inner_dims = int(in_dims//4)
        # main branch
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True)
        )

        # global pooling branch
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.pool_branch = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True)
        )

        # feature pyramid branch
        self.conv7x7_1 = nn.Sequential(
            nn.Conv2d(in_dims, inner_dims, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=inner_dims),
            nn.ReLU(inplace=True)
        )
        self.conv5x5_1 = nn.Sequential(
            nn.Conv2d(inner_dims, inner_dims, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(num_features=inner_dims),
            nn.ReLU(inplace=True)
        )
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(inner_dims, inner_dims, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=inner_dims),
            nn.ReLU(inplace=True)
        )

        self.conv7x7_2 = nn.Sequential(
            nn.Conv2d(inner_dims, inner_dims, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(num_features=inner_dims),
            nn.ReLU(inplace=True)
        )
        self.conv5x5_2 = nn.Sequential(
            nn.Conv2d(inner_dims, inner_dims, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(num_features=inner_dims),
            nn.ReLU(inplace=True)
        )
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(inner_dims, inner_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=inner_dims),
            nn.ReLU(inplace=True)
        )

        self.conv7x7_up = nn.Sequential(
            nn.ConvTranspose2d(inner_dims, out_dims, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True)
        )
        self.conv5x5_up = nn.Sequential(
            nn.ConvTranspose2d(inner_dims, inner_dims, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=inner_dims),
            nn.ReLU(inplace=True)
        )
        self.conv3x3_up = nn.Sequential(
            nn.ConvTranspose2d(inner_dims, inner_dims, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=inner_dims),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # main branch
        out_main = self.main_branch(x)

        # global pooling branch
        out_pool = self.pool(x)
        out_pool = self.pool_branch(out_pool)

        # 7x7 branch
        out_conv7x7_1 = self.conv7x7_1(x)
        out_conv7x7_2 = self.conv7x7_2(out_conv7x7_1)

        # 5x5 branch
        out_conv5x5_1 = self.conv5x5_1(out_conv7x7_1)
        out_conv5x5_2 = self.conv5x5_2(out_conv5x5_1)

        # 3x3 branch
        out_conv3x3_1 = self.conv3x3_1(out_conv5x5_1)
        out_conv3x3_2 = self.conv3x3_2(out_conv3x3_1)

        # upsample
        out_conv3x3_up = self.conv3x3_up(out_conv3x3_2)
        out_conv5x5_merge = self.relu(out_conv5x5_2 + out_conv3x3_up)

        out_conv5x5_up = self.conv5x5_up(out_conv5x5_merge)
        out_conv7x7_merge = self.relu(out_conv7x7_2 + out_conv5x5_up)
        out_conv7x7_up = self.conv7x7_up(out_conv7x7_merge)

        # merge
        out_merge = out_main * out_conv7x7_up
        out_merge = self.relu(out_merge + out_pool)

        return out_merge


class GlobalAttentionUpsample(nn.Module):
    def __init__(self, high_level_dims, low_level_dims):
        super(GlobalAttentionUpsample, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_level_dims, low_level_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=low_level_dims),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_level_dims, low_level_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=low_level_dims),
            nn.ReLU(inplace=True)
        )

        self.up = nn.Sequential(
            nn.Conv2d(high_level_dims, low_level_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=low_level_dims),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_high, x_low):
        out_low = self.conv_low(x_low)

        out_high = self.pool(x_high)
        out_high = self.conv_high(out_high)

        out_merge = self.relu(out_low * out_high)

        out_high_up = F.interpolate(out_high, size=x_low.shape[-2:], mode='bilinear', align_corners=True)
        out_merge = out_merge + self.up(out_high_up)

        out_merge = self.relu(out_merge)

        return out_merge


class PyramidAttentionNetwork(nn.Module):
    def __init__(self, in_channels, num_classes, model='resnet50'):
        super(PyramidAttentionNetwork, self).__init__()

        if model == 'resnet50':
            self.backbone = resnet50(in_channels=in_channels)
        elif model == 'resnet101':
            self.backbone = resnet101(in_channels=in_channels)
        elif model == 'resnet152':
            self.backbone = resnet152(in_channels=in_channels)

        self.fpa = FeaturePyramidAttention(in_dims=2048, out_dims=256)
        self.gau = GlobalAttentionUpsample(high_level_dims=256, low_level_dims=256)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out, low_level_features = self.backbone(x)
        out_fpa = self.fpa(out)
        out_gau = self.gau(low_level_features, out_fpa)
        out = self.classifier(out_gau)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)

        return out


def pan_resnet50(**kwargs):
    model = PyramidAttentionNetwork(model='resnet50', **kwargs)

    return model


def pan_resnet101(**kwargs):
    model = PyramidAttentionNetwork(model='resnet101', **kwargs)

    return model


def pan_resnet152(**kwargs):
    model = PyramidAttentionNetwork(model='resnet152', **kwargs)

    return model


if __name__ == '__main__':
    img_size = 512
    model = pan_resnet101(in_channels=3, num_classes=1000)

    input = torch.randn(4, 3, img_size, img_size)
    output = model(input)

    print(output.shape)