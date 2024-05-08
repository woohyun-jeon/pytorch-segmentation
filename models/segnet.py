import torch
import torch.nn as nn


__all__ = ['SegNet', 'segnet']


# set SegNet configuration
cfgs = [64, 128, 256, 512]


class down_2_block(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(down_2_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dims, out_channels=out_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_dims, out_channels=out_dims, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool_with_argmax = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.conv(x)
        unpooled_shape = x.shape
        x, indices = self.maxpool_with_argmax(x)

        return x, indices, unpooled_shape


class down_3_block(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(down_3_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dims, out_channels=out_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_dims, out_channels=out_dims, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_dims, out_channels=out_dims, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool_with_argmax = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.conv(x)
        unpooled_shape = x.shape
        x, indices = self.maxpool_with_argmax(x)

        return x, indices, unpooled_shape


class up_2_block(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(up_2_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dims, out_channels=out_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_dims, out_channels=out_dims, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x, unpool_indices, unpool_shape):
        x = self.unpool(x, indices=unpool_indices, output_size=unpool_shape)
        x = self.conv(x)

        return x


class up_3_block(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(up_3_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dims, out_channels=out_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_dims, out_channels=out_dims, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_dims, out_channels=out_dims, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x, unpool_indices, unpool_shape):
        x = self.unpool(x, indices=unpool_indices, output_size=unpool_shape)
        x = self.conv(x)

        return x


class SegNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegNet, self).__init__()
        # encoding layers
        self.down1 = down_2_block(in_channels, cfgs[0])
        self.down2 = down_2_block(cfgs[0], cfgs[1])
        self.down3 = down_3_block(cfgs[1], cfgs[2])
        self.down4 = down_3_block(cfgs[2], cfgs[3])
        self.down5 = down_3_block(cfgs[3], cfgs[3])

        # decoding layers
        self.up5 = up_3_block(cfgs[3], cfgs[3])
        self.up4 = up_3_block(cfgs[3], cfgs[2])
        self.up3 = up_3_block(cfgs[2], cfgs[1])
        self.up2 = up_2_block(cfgs[1], cfgs[0])
        self.up1 = up_2_block(cfgs[0], num_classes)

    def forward(self, x):
        down1, unpool_indices1, unpool_shape1 = self.down1(x)
        down2, unpool_indices2, unpool_shape2 = self.down2(down1)
        down3, unpool_indices3, unpool_shape3 = self.down3(down2)
        down4, unpool_indices4, unpool_shape4 = self.down4(down3)
        down5, unpool_indices5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, unpool_indices5, unpool_shape5)
        up4 = self.up4(up5, unpool_indices4, unpool_shape4)
        up3 = self.up3(up4, unpool_indices3, unpool_shape3)
        up2 = self.up2(up3, unpool_indices2, unpool_shape2)
        up1 = self.up1(up2, unpool_indices1, unpool_shape1)

        return up1


def segnet(**kwargs):
    model = SegNet(**kwargs)

    return model


if __name__ == '__main__':
    img_size = 512
    model = segnet(in_channels=3, num_classes=1000)

    input = torch.randn(4, 3, img_size, img_size)
    output = model(input)

    print(output.shape)