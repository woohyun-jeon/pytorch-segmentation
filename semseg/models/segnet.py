import torch.nn as nn

class down_2_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down_2_block, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)
    def forward(self, x):
        x = self.conv(x)
        unpooled_shape = x.shape
        x, indices = self.maxpool_with_argmax(x)
        return x, indices, unpooled_shape

class down_3_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down_3_block, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)
    def forward(self, x):
        x = self.conv(x)
        unpooled_shape = x.shape
        x, indices = self.maxpool_with_argmax(x)
        return x, indices, unpooled_shape

class up_2_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_2_block, self).__init__()
        self.in_channels = in_channels
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, unpool_indices, unpool_shape):
        x = self.unpool(x, indices=unpool_indices, output_size=unpool_shape)
        x = self.conv(x)
        return x

class up_3_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_3_block, self).__init__()
        self.in_channels = in_channels
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, unpool_indices, unpool_shape):
        x = self.unpool(x, indices=unpool_indices, output_size=unpool_shape)
        x = self.conv(x)
        return x

class SegNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # encoding layers
        self.down1 = down_2_block(self.in_channels, 64)
        self.down2 = down_2_block(64, 128)
        self.down3 = down_3_block(128, 256)
        self.down4 = down_3_block(256, 512)
        self.down5 = down_3_block(512, 512)

        # decoding layers
        self.up5 = up_3_block(512, 512)
        self.up4 = up_3_block(512, 256)
        self.up3 = up_3_block(256, 128)
        self.up2 = up_2_block(128, 64)
        self.up1 = up_2_block(64, self.num_classes)

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

# import torch
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# model = SegNet(in_channels=3, num_classes=10).to(device)
# print(model)