import torch
import torch.nn as nn


__all__ = ['UnetPlusPlus', 'unetplusplus']


# set Unet++ configuration
cfgs = [64, 128, 256, 512, 1024]


class UnetBlock(nn.Module):
    def __init__(self, in_dims, mid_dims, out_dims):
        super(UnetBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dims, mid_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=mid_dims),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_dims, out_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class UnetPlusPlus(nn.Module):
    def __init__(self, in_channels, num_classes, deep_supervision=False):
        super(UnetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision
        # encoding
        self.conv0_0 = UnetBlock(in_channels, cfgs[0], cfgs[0])
        self.conv1_0 = UnetBlock(cfgs[0], cfgs[1], cfgs[1])
        self.conv2_0 = UnetBlock(cfgs[1], cfgs[2], cfgs[2])
        self.conv3_0 = UnetBlock(cfgs[2], cfgs[3], cfgs[3])
        self.conv4_0 = UnetBlock(cfgs[3], cfgs[4], cfgs[4])

        # decoding
        self.conv0_1 = UnetBlock(cfgs[0]+cfgs[1], cfgs[0], cfgs[0])
        self.conv1_1 = UnetBlock(cfgs[1]+cfgs[2], cfgs[1], cfgs[1])
        self.conv2_1 = UnetBlock(cfgs[2]+cfgs[3], cfgs[2], cfgs[2])
        self.conv3_1 = UnetBlock(cfgs[3]+cfgs[4], cfgs[3], cfgs[3])

        self.conv0_2 = UnetBlock(cfgs[0]*2+cfgs[1], cfgs[0], cfgs[0])
        self.conv1_2 = UnetBlock(cfgs[1]*2+cfgs[2], cfgs[1], cfgs[1])
        self.conv2_2 = UnetBlock(cfgs[2]*2+cfgs[3], cfgs[2], cfgs[2])

        self.conv0_3 = UnetBlock(cfgs[0]*3+cfgs[1], cfgs[0], cfgs[0])
        self.conv1_3 = UnetBlock(cfgs[1]*3+cfgs[2], cfgs[1], cfgs[1])

        self.conv0_4 = UnetBlock(cfgs[0]*4+cfgs[1], cfgs[0], cfgs[0])

        if self.deep_supervision:
            self.out1 = nn.Conv2d(cfgs[0], num_classes, kernel_size=1, stride=1, padding=0)
            self.out2 = nn.Conv2d(cfgs[0], num_classes, kernel_size=1, stride=1, padding=0)
            self.out3 = nn.Conv2d(cfgs[0], num_classes, kernel_size=1, stride=1, padding=0)
            self.out4 = nn.Conv2d(cfgs[0], num_classes, kernel_size=1, stride=1, padding=0)
        else:
            self.out = nn.Conv2d(cfgs[0], num_classes, kernel_size=1, stride=1, padding=0)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(self.maxpool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))

        x2_0 = self.conv2_0(self.maxpool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.maxpool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.maxpool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        if self.deep_supervision:
            out1 = self.out1(x0_1)
            out2 = self.out2(x0_2)
            out3 = self.out3(x0_3)
            out4 = self.out4(x0_4)
            out = (out1 + out2 + out3 + out4) / 4
        else:
            out = self.out(x0_4)

        return out


def unetplusplus(**kwargs):
    model = UnetPlusPlus(**kwargs)

    return model


if __name__ == '__main__':
    img_size = 512
    model = unetplusplus(in_channels=3, num_classes=1000)

    input = torch.randn(1, 3, img_size, img_size)
    output = model(input)

    print(output.shape)