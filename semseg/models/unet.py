import torch
import torch.nn as nn


__all__ = ['Unet', 'unet']


# set Unet configuration
cfgs = [64, 128, 256, 512, 1024]


class UnetBlock(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(UnetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class Unet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Unet, self).__init__()
        self.dims = [in_channels] + cfgs

        # contracting path
        self.enc1_1 = UnetBlock(in_channels, cfgs[0])
        self.enc1_2 = UnetBlock(cfgs[0], cfgs[0])
        self.enc2_1 = UnetBlock(cfgs[0], cfgs[1])
        self.enc2_2 = UnetBlock(cfgs[1], cfgs[1])
        self.enc3_1 = UnetBlock(cfgs[1], cfgs[2])
        self.enc3_2 = UnetBlock(cfgs[2], cfgs[2])
        self.enc4_1 = UnetBlock(cfgs[2], cfgs[3])
        self.enc4_2 = UnetBlock(cfgs[3], cfgs[3])

        # bridge
        self.bridge_1 = UnetBlock(cfgs[3], cfgs[4])
        self.bridge_2 = UnetBlock(cfgs[4], cfgs[4])

        # expansive path
        self.up_4 = nn.ConvTranspose2d(cfgs[4], cfgs[3], kernel_size=2, stride=2)
        self.dec4_1 = UnetBlock(cfgs[4], cfgs[3])
        self.dec4_2 = UnetBlock(cfgs[3], cfgs[3])

        self.up_3 = nn.ConvTranspose2d(cfgs[3], cfgs[2], kernel_size=2, stride=2)
        self.dec3_1 = UnetBlock(cfgs[3], cfgs[2])
        self.dec3_2 = UnetBlock(cfgs[2], cfgs[2])

        self.up_2 = nn.ConvTranspose2d(cfgs[2], cfgs[1], kernel_size=2, stride=2)
        self.dec2_1 = UnetBlock(cfgs[2], cfgs[1])
        self.dec2_2 = UnetBlock(cfgs[1], cfgs[1])

        self.up_1 = nn.ConvTranspose2d(cfgs[1], cfgs[0], kernel_size=2, stride=2)
        self.dec1_1 = UnetBlock(cfgs[1], cfgs[0])
        self.dec1_2 = UnetBlock(cfgs[0], cfgs[0])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # set classifier
        self.classifier = nn.Conv2d(in_channels=cfgs[0], out_channels=num_classes,
                                    kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # encoding
        x_enc1_1 = self.enc1_1(x)
        x_enc1_2 = self.enc1_2(x_enc1_1)
        x_enc1_out = self.maxpool(x_enc1_2)

        x_enc2_1 = self.enc2_1(x_enc1_out)
        x_enc2_2 = self.enc2_2(x_enc2_1)
        x_enc2_out = self.maxpool(x_enc2_2)

        x_enc3_1 = self.enc3_1(x_enc2_out)
        x_enc3_2 = self.enc3_2(x_enc3_1)
        x_enc3_out = self.maxpool(x_enc3_2)

        x_enc4_1 = self.enc4_1(x_enc3_out)
        x_enc4_2 = self.enc4_2(x_enc4_1)
        x_enc4_out = self.maxpool(x_enc4_2)

        # bridge
        x_bridge_1 = self.bridge_1(x_enc4_out)
        x_bridge_2 = self.bridge_2(x_bridge_1)

        # decoding
        x_dec4_in = torch.cat([self.up_4(x_bridge_2), x_enc4_2], dim=1)
        x_dec4_1 = self.dec4_1(x_dec4_in)
        x_dec4_2 = self.dec4_2(x_dec4_1)

        x_dec3_in = torch.cat([self.up_3(x_dec4_2), x_enc3_2], dim=1)
        x_dec3_1 = self.dec3_1(x_dec3_in)
        x_dec3_2 = self.dec3_2(x_dec3_1)

        x_dec2_in = torch.cat([self.up_2(x_dec3_2), x_enc2_2], dim=1)
        x_dec2_1 = self.dec2_1(x_dec2_in)
        x_dec2_2 = self.dec2_2(x_dec2_1)

        x_dec1_in = torch.cat([self.up_1(x_dec2_2), x_enc1_2], dim=1)
        x_dec1_1 = self.dec1_1(x_dec1_in)
        x_dec1_2 = self.dec1_2(x_dec1_1)

        out = self.classifier(x_dec1_2)

        return out


def unet(**kwargs):
    model = Unet(**kwargs)

    return model


if __name__ == '__main__':
    img_size = 512
    model = unet(in_channels=3, num_classes=1000)

    input = torch.randn(1, 3, img_size, img_size)
    output = model(input)

    print(output.shape)