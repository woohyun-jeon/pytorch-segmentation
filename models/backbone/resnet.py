import torch
import torch.nn as nn


__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']


# set ResNet configuration
cfgs = [64, 128, 256, 512]


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_dims, out_dims, dilation=1, stride=1):
        super(Bottleneck, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_dims, out_dims, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_dims, out_dims*Bottleneck.expansion, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_dims*Bottleneck.expansion)
        )

        if stride != 1 or in_dims != out_dims*Bottleneck.expansion:
            self.shortcut = nn.Conv2d(in_dims, out_dims*Bottleneck.expansion, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.residual(x) + self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels, output_stride=8):
        super(ResNet, self).__init__()
        stride_lists = None
        atrous_rates = None
        if output_stride == 8:
            stride_lists = [1,2,1,1]
            atrous_rates = [1,1,2,4]
        elif output_stride == 16:
            stride_lists = [1,2,2,1]
            atrous_rates = [1,1,1,2]

        self.in_channels = cfgs[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, cfgs[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=cfgs[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layers(block, num_blocks[0], cfgs[0], stride=stride_lists[0], dilation=atrous_rates[0])
        self.conv3_x = self._make_layers(block, num_blocks[1], cfgs[1], stride=stride_lists[1], dilation=atrous_rates[1])
        self.conv4_x = self._make_layers(block, num_blocks[2], cfgs[2], stride=stride_lists[2], dilation=atrous_rates[2])
        self.conv5_x = self._make_layers(block, num_blocks[3], cfgs[3], stride=stride_lists[3], dilation=atrous_rates[3])

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.conv2_x(out)
        low_level_feature = out
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)

        return out, low_level_feature

    def _make_layers(self, block, num_blocks, out_dims, stride, dilation):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_dims=self.in_channels, out_dims=out_dims, dilation=dilation, stride=stride))
            self.in_channels = out_dims*block.expansion

        return nn.Sequential(*layers)


def resnet50(**kwargs):
    return ResNet(block=Bottleneck, num_blocks=[3,4,6,3], **kwargs)


def resnet101(**kwargs):
    return ResNet(block=Bottleneck, num_blocks=[3,4,23,3], **kwargs)


def resnet152(**kwargs):
    return ResNet(block=Bottleneck, num_blocks=[3,8,36,3], **kwargs)


if __name__ == '__main__':
    img_size = 224

    model = resnet101(in_channels=3)

    input = torch.randn(4, 3, img_size, img_size)

    output, low_level_feature = model(input)
    print(output.shape, low_level_feature.shape)