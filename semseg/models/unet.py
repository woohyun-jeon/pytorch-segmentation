import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# set UNet configuration
cfgs = [64, 128, 256, 512, 1024]


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        # set contracting path
        self.dims = [in_channels] + cfgs
        self.encoding = nn.ModuleList(
            [self.unet_parts(in_dim, out_dim) for in_dim, out_dim in zip(self.dims[:-1], self.dims[1:])]
        )
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        # set expansive path
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
             for in_dim, out_dim in zip(self.dims[::-1][:-2], self.dims[::-1][1:-1])]
        )
        self.decoding = nn.ModuleList(
            [self.unet_parts(in_dim, out_dim) for in_dim, out_dim in zip(self.dims[::-1][:-2], self.dims[::-1][1:-1])]
        )

        # set classifier
        self.classifier = nn.Conv2d(in_channels=cfgs[0], out_channels=num_classes,
                                    kernel_size=1, stride=1, padding=0, bias=False)

    def unet_parts(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        return conv

    def forward(self, x):
        # do encoding
        down_layers = []
        for enc_layer in self.encoding:
            x = enc_layer(x)
            if enc_layer is not self.encoding[-1]:
                down_layers.append(x)
                x = self.down(x)

        down_layers = down_layers[::-1]

        # do decoding
        for up_layer, dec_layer, concat_layer in zip(self.up, self.decoding, down_layers):
            x = up_layer(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layer.shape[2:])

            x_concat = torch.cat((concat_layer, x), dim=1)
            x = dec_layer(x_concat)

        # do classifier
        x = self.classifier(x)

        return x