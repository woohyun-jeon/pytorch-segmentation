import torch
import torch.nn as nn
from .backbone.swin_transformer import SwinEncoderBlock


__all__ = ['SwinUnet', 'swin_unet']


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embedding_dims, patch_size):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, embedding_dims, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        out = self.conv(x)
        out = out.permute(0, 2, 3, 1)

        return out


class PatchMerging(nn.Module):
    def __init__(self, embedding_dims):
        super(PatchMerging, self).__init__()
        self.norm = nn.LayerNorm(4 * embedding_dims, eps=1e-6)
        self.reduction = nn.Linear(4 * embedding_dims, 2 * embedding_dims, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, int(H / 2), 2, int(W / 2), 2, C).permute(0, 1, 3, 4, 2, 5) # (B, H/2, W/2, 2, 2, C)
        x = torch.flatten(x, 3) # (B, H/2, W/2, 4C)
        x = self.norm(x) # (B, H/2, W/2, 4C)
        x = self.reduction(x) # (B, H/2, W/2, 2C)
        return x


class PatchExpansion(nn.Module):
    def __init__(self, embedding_dims):
        super(PatchExpansion, self).__init__()
        self.norm = nn.LayerNorm(int(embedding_dims / 2), eps=1e-6)
        self.expand = nn.Linear(embedding_dims, embedding_dims * 2, bias=False)

    def forward(self, x):
        x = self.expand(x) # (B, H, W, 2C)

        B, H, W, C = x.shape# (B, H, W, 2C)
        x = x.reshape(B, H, W, 2, 2, int(C / 4)).permute(0, 1, 3, 2, 4, 5) # (B, H, 2, W, 2, C//2)
        x = x.reshape(B, H*2, W*2, int(C / 4))
        x = self.norm(x) # (B, 2*H, 2*W, C/2)

        return x


class LastExpansion(nn.Module):
    def __init__(self, embedding_dims):
        super(LastExpansion, self).__init__()
        self.norm = nn.LayerNorm(embedding_dims, eps=1e-6)
        self.expand = nn.Linear(embedding_dims, embedding_dims * 16, bias=False)

    def forward(self, x):
        x = self.expand(x) # (B, H, W, 16C)

        B, H, W, C = x.shape# (B, H, W, 16C)
        x = x.reshape(B, H, W, 4, 4, int(C / 16)).permute(0, 1, 3, 2, 4, 5) # (B, H, 4, W, 4, C/16)
        x = x.reshape(B, H*4, W*4, int(C / 16))
        x = self.norm(x) # (B, 4*H, 4*W, C)

        return x


class SwinBlock(nn.Module):
    def __init__(self, embedding_dims, num_heads, window_size=7, p_dropout=0.5):
        super(SwinBlock, self).__init__()
        self.swa = SwinEncoderBlock(shifted=False, relative_pos_embedding=True, embedding_dims=embedding_dims,
                                    num_heads=num_heads, window_size=window_size,p_dropout=p_dropout)
        self.mswa = SwinEncoderBlock(shifted=True, relative_pos_embedding=True, embedding_dims=embedding_dims,
                                     num_heads=num_heads, window_size=window_size,p_dropout=p_dropout)

    def forward(self, x):
        x = self.swa(x)
        x = self.mswa(x)

        return x


class SwinUnet(nn.Module):
    def __init__(self, in_channels, num_classes, embedding_dims, patch_size=4):
        super(SwinUnet, self).__init__()
        # contracting path
        self.enc1_1 = PatchEmbedding(in_channels=in_channels, embedding_dims=embedding_dims*1, patch_size=patch_size)
        self.enc1_2 = SwinBlock(embedding_dims=embedding_dims*1, num_heads=3, window_size=7, p_dropout=0.5)
        self.enc2_1 = PatchMerging(embedding_dims=embedding_dims*1)
        self.enc2_2 = SwinBlock(embedding_dims=embedding_dims*2, num_heads=3, window_size=7, p_dropout=0.5)
        self.enc3_1 = PatchMerging(embedding_dims=embedding_dims*2)
        self.enc3_2 = SwinBlock(embedding_dims=embedding_dims*4, num_heads=3, window_size=7, p_dropout=0.5)
        self.enc4_1 = PatchMerging(embedding_dims=embedding_dims*4)

        # bridge
        self.bridge_1 = SwinBlock(embedding_dims=embedding_dims*8, num_heads=3, window_size=7, p_dropout=0.5)
        self.bridge_2 = SwinBlock(embedding_dims=embedding_dims*8, num_heads=3, window_size=7, p_dropout=0.5)

        # expanding path
        self.dec1_1 = PatchExpansion(embedding_dims=embedding_dims*8)
        self.dec1_2 = SwinBlock(embedding_dims=embedding_dims*8, num_heads=3, window_size=7, p_dropout=0.5)
        self.dec2_1 = PatchExpansion(embedding_dims=embedding_dims*4)
        self.dec2_2 = SwinBlock(embedding_dims=embedding_dims*4, num_heads=3, window_size=7, p_dropout=0.5)
        self.dec3_1 = PatchExpansion(embedding_dims=embedding_dims*2)
        self.dec3_2 = SwinBlock(embedding_dims=embedding_dims*2, num_heads=3, window_size=7, p_dropout=0.5)
        self.last = LastExpansion(embedding_dims=embedding_dims*1)

        # merge path
        self.skip1_1 = nn.Linear(embedding_dims*8, embedding_dims*4)
        self.skip2_1 = nn.Linear(embedding_dims*4, embedding_dims*2)
        self.skip3_1 = nn.Linear(embedding_dims*2, embedding_dims*1)

        self.classifier = nn.Conv2d(embedding_dims, num_classes, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # encoding
        x_enc1_1 = self.enc1_1(x)
        x_enc1_2 = self.enc1_2(x_enc1_1)
        x_enc2_1 = self.enc2_1(x_enc1_2)
        x_enc2_2 = self.enc2_2(x_enc2_1)
        x_enc3_1 = self.enc3_1(x_enc2_2)
        x_enc3_2 = self.enc3_2(x_enc3_1)
        x_enc4_1 = self.enc4_1(x_enc3_2)

        # bridge
        x_bridge_1 = self.bridge_1(x_enc4_1)
        x_bridge_2 = self.bridge_2(x_bridge_1)

        # decoding
        x_dec1_1 = self.dec1_1(x_bridge_2)
        x_dec1_2 = self.dec1_2(torch.cat([x_dec1_1, x_enc3_2], dim=-1))
        x_dec2_1 = self.dec2_1(self.skip1_1(x_dec1_2))
        x_dec2_2 = self.dec2_2(torch.cat([x_dec2_1, x_enc2_2], dim=-1))
        x_dec3_1 = self.dec3_1(self.skip2_1(x_dec2_2))
        x_dec3_2 = self.dec3_2(torch.cat([x_dec3_1, x_enc1_2], dim=-1))
        x_last = self.last(self.skip3_1(x_dec3_2))

        out = self.classifier(x_last.permute(0, 3, 1, 2))

        return out


def swin_unet(**kwargs):
    return SwinUnet(embedding_dims=96, patch_size=4, **kwargs)


if __name__ == '__main__':
    img_size = 224
    in_channels = 3

    input = torch.randn(8, in_channels, img_size, img_size)

    model = swin_unet(in_channels=in_channels, num_classes=1000)
    out = model(input)
    print(out.shape)