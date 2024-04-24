import torch
import torch.nn as nn


__all__ = ['Unet3Plus', 'unet3plus']


# set Unet3+ configuration
cfgs = [64, 128, 256, 512, 1024]


class UnetBlock(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(UnetBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_dims, out_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class Unet3Plus(nn.Module):
    def __init__(self, in_channels, num_classes, deep_supervision=False):
        super(Unet3Plus, self).__init__()
        self.deep_supervision = deep_supervision

        # encoding
        self.enc1 = UnetBlock(in_channels, cfgs[0])
        self.enc2 = UnetBlock(cfgs[0], cfgs[1])
        self.enc3 = UnetBlock(cfgs[1], cfgs[2])
        self.enc4 = UnetBlock(cfgs[2], cfgs[3])
        self.enc5 = UnetBlock(cfgs[3], cfgs[4])

        # decoding
        self.enc4_dec4 = UnetBlock(cfgs[3], cfgs[0])
        self.enc3_dec4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            UnetBlock(cfgs[2], cfgs[0])
        )
        self.enc2_dec4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            UnetBlock(cfgs[1], cfgs[0])
        )
        self.enc1_dec4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            UnetBlock(cfgs[0], cfgs[0])
        )
        self.up_enc5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            UnetBlock(cfgs[4], cfgs[0])
        )

        self.enc3_dec3 = UnetBlock(cfgs[2], cfgs[0])
        self.enc2_dec3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            UnetBlock(cfgs[1], cfgs[0])
        )
        self.enc1_dec3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            UnetBlock(cfgs[0], cfgs[0])
        )
        self.enc5_dec3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            UnetBlock(cfgs[4], cfgs[0])
        )
        self.up_dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            UnetBlock(5*cfgs[0], cfgs[0])
        )

        self.enc2_dec2 = UnetBlock(cfgs[1], cfgs[0])
        self.enc1_dec2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            UnetBlock(cfgs[0], cfgs[0])
        )
        self.enc5_dec2 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            UnetBlock(cfgs[4], cfgs[0])
        )
        self.dec4_dec2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            UnetBlock(5*cfgs[0], cfgs[0])
        )
        self.up_dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            UnetBlock(5*cfgs[0], cfgs[0])
        )

        self.enc1_dec1 = UnetBlock(cfgs[0], cfgs[0])
        self.enc5_dec1 = nn.Sequential(
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
            UnetBlock(cfgs[4], cfgs[0])
        )
        self.dec4_dec1 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            UnetBlock(5*cfgs[0], cfgs[0])
        )
        self.dec3_dec1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            UnetBlock(5*cfgs[0], cfgs[0])
        )
        self.up_dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            UnetBlock(5*cfgs[0], cfgs[0])
        )

        self.conv = UnetBlock(5*cfgs[0], 5*cfgs[0])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.deep_supervision:
            self.out1 = nn.Sequential(
                nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
                nn.Conv2d(cfgs[4], num_classes, kernel_size=1, stride=1, padding=0)
            )
            self.out2 = nn.Sequential(
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
                nn.Conv2d(5*cfgs[0], num_classes, kernel_size=1, stride=1, padding=0)
            )
            self.out3 = nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                nn.Conv2d(5*cfgs[0], num_classes, kernel_size=1, stride=1, padding=0)
            )
            self.out4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(5*cfgs[0], num_classes, kernel_size=1, stride=1, padding=0)
            )
            self.out5 = nn.Conv2d(5*cfgs[0], num_classes, kernel_size=1, stride=1, padding=0)
        else:
            self.out = nn.Conv2d(5*cfgs[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_enc1 = self.enc1(x)
        x_enc2 = self.maxpool(self.enc2(x_enc1))
        x_enc3 = self.maxpool(self.enc3(x_enc2))
        x_enc4 = self.maxpool(self.enc4(x_enc3))
        x_enc5 = self.maxpool(self.enc5(x_enc4))

        x_dec4 = self.conv(torch.cat([self.enc1_dec4(x_enc1), self.enc2_dec4(x_enc2), self.enc3_dec4(x_enc3),
                                      self.enc4_dec4(x_enc4), self.up_enc5(x_enc5)], dim=1))
        x_dec3 = self.conv(torch.cat([self.enc1_dec3(x_enc1), self.enc2_dec3(x_enc2), self.enc3_dec3(x_enc3),
                                      self.up_dec4(x_dec4), self.enc5_dec3(x_enc5)], dim=1))
        x_dec2 = self.conv(torch.cat([self.enc1_dec2(x_enc1), self.enc2_dec2(x_enc2), self.up_dec3(x_dec3),
                                      self.enc5_dec2(x_enc5), self.dec4_dec2(x_dec4)], dim=1))
        x_dec1 = self.conv(torch.cat([self.enc1_dec1(x_enc1), self.up_dec2(x_dec2), self.enc5_dec1(x_enc5),
                                      self.dec4_dec1(x_dec4), self.dec3_dec1(x_dec3)], dim=1))

        if self.deep_supervision:
            out1 = self.out1(x_enc5)
            out2 = self.out2(x_dec4)
            out3 = self.out3(x_dec3)
            out4 = self.out4(x_dec2)
            out5 = self.out5(x_dec1)
            out = (out1 + out2 + out3 + out4 + out5) / 5
        else:
            out = self.out(x_dec1)

        return out


def unet3plus(**kwargs):
    model = Unet3Plus(**kwargs)

    return model


if __name__ == '__main__':
    img_size = 512
    model = unet3plus(in_channels=3, num_classes=1000)

    input = torch.randn(1, 3, img_size, img_size)
    output = model(input)

    print(output.shape)