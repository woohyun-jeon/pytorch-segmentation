import torch
import torch.nn as nn


__all__ = ['AttentionUNet', 'attention_unet']


# set Attention Unet configuration
cfgs = [64, 128, 256, 512, 1024]


class UnetBlock(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(UnetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(UpConvBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.up(x)

        return out


class AttentionGate(nn.Module):
    def __init__(self, in_dims_g, in_dims_x, out_dims):
        super(AttentionGate, self).__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(in_dims_g, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_dims),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(in_dims_x, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_dims),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(out_dims, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g_out = self.w_g(g)
        x_out = self.w_x(x)
        psi_out = self.relu(g_out + x_out)
        psi_out = self.psi(psi_out)
        out = psi_out * x

        return out


class AttentionUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AttentionUNet, self).__init__()
        # contracting path
        self.enc1_1 = UnetBlock(in_channels, cfgs[0])
        self.enc1_2 = UnetBlock(cfgs[0], cfgs[0])
        self.enc2_1 = UnetBlock(cfgs[0], cfgs[1])
        self.enc2_2 = UnetBlock(cfgs[1], cfgs[1])
        self.enc3_1 = UnetBlock(cfgs[1], cfgs[2])
        self.enc3_2 = UnetBlock(cfgs[2], cfgs[2])
        self.enc4_1 = UnetBlock(cfgs[2], cfgs[3])
        self.enc4_2 = UnetBlock(cfgs[3], cfgs[3])
        self.enc5_1 = UnetBlock(cfgs[3], cfgs[4])
        self.enc5_2 = UnetBlock(cfgs[4], cfgs[4])

        # expansive path
        self.up_5 = UpConvBlock(cfgs[4], cfgs[3])
        self.att_5 = AttentionGate(cfgs[3], cfgs[3], cfgs[2])
        self.dec5_1 = UnetBlock(cfgs[4], cfgs[3])
        self.dec5_2 = UnetBlock(cfgs[3], cfgs[3])

        self.up_4 = UpConvBlock(cfgs[3], cfgs[2])
        self.att_4 = AttentionGate(cfgs[2], cfgs[2], cfgs[1])
        self.dec4_1 = UnetBlock(cfgs[3], cfgs[2])
        self.dec4_2 = UnetBlock(cfgs[2], cfgs[2])

        self.up_3 = UpConvBlock(cfgs[2], cfgs[1])
        self.att_3 = AttentionGate(cfgs[1], cfgs[1], cfgs[0])
        self.dec3_1 = UnetBlock(cfgs[2], cfgs[1])
        self.dec3_2 = UnetBlock(cfgs[1], cfgs[1])

        self.up_2 = UpConvBlock(cfgs[1], cfgs[0])
        self.att_2 = AttentionGate(cfgs[0], cfgs[0], int(cfgs[0]//2))
        self.dec2_1 = UnetBlock(cfgs[1], cfgs[0])
        self.dec2_2 = UnetBlock(cfgs[0], cfgs[0])

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

        x_enc5_1 = self.enc5_1(x_enc4_out)
        x_enc5_2 = self.enc5_2(x_enc5_1)

        # decoding
        x_dec5_1 = self.up_5(x_enc5_2)
        x_dec5_2 = self.att_5(g=x_dec5_1, x=x_enc4_2)
        x_dec5_in = torch.cat((x_dec5_1, x_dec5_2), dim=1)
        x_dec5_out = self.dec5_1(x_dec5_in)
        x_dec5_out = self.dec5_2(x_dec5_out)

        x_dec4_1 = self.up_4(x_dec5_out)
        x_dec4_2 = self.att_4(g=x_dec4_1, x=x_enc3_2)
        x_dec4_in = torch.cat((x_dec4_1, x_dec4_2), dim=1)
        x_dec4_out = self.dec4_1(x_dec4_in)
        x_dec4_out = self.dec4_2(x_dec4_out)

        x_dec3_1 = self.up_3(x_dec4_out)
        x_dec3_2 = self.att_3(g=x_dec3_1, x=x_enc2_2)
        x_dec3_in = torch.cat((x_dec3_1, x_dec3_2), dim=1)
        x_dec3_out = self.dec3_1(x_dec3_in)
        x_dec3_out = self.dec3_2(x_dec3_out)

        x_dec2_1 = self.up_2(x_dec3_out)
        x_dec2_2 = self.att_2(g=x_dec2_1, x=x_enc1_2)
        x_dec2_in = torch.cat((x_dec2_1, x_dec2_2), dim=1)
        x_dec2_out = self.dec2_1(x_dec2_in)
        x_dec2_out = self.dec2_2(x_dec2_out)

        out = self.classifier(x_dec2_out)

        return out


def attention_unet(**kwargs):
    return AttentionUNet(**kwargs)


if __name__ == '__main__':
    img_size = 512
    model = attention_unet(in_channels=3, num_classes=1000)

    input = torch.randn(1, 3, img_size, img_size)
    output = model(input)

    print(output.shape)