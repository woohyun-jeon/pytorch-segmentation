# [Reference] https://github.com/berniwal/swin-transformer-pytorch/blob/master/swin_transformer_pytorch/swin_transformer.py

import math
import numpy as np

import torch
import torch.nn as nn
from einops import rearrange


__all__ = ['SwinEncoderBlock', 'SwinTransformer', 'swin_t', 'swin_s', 'swin_b', 'swin_l']


def get_mask(window_size, shift_size, up_down, left_right):
    mask = torch.zeros(size=(window_size ** 2, window_size ** 2))

    if up_down:
        mask[-(shift_size * window_size):, :-(shift_size * window_size)] = float('-inf')
        mask[:-(shift_size * window_size), -(shift_size * window_size):] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -shift_size:, :, :-shift_size] = float('-inf')
        mask[:, :-shift_size, :, -shift_size:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x,y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]

    return distances


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dims=96, num_heads=8, window_size=7, p_dropout=0.5, shifted=False, relative_pos_embedding=False):
        super(MultiHeadSelfAttention, self).__init__()
        self.window_size = window_size

        self.shifted = shifted
        if self.shifted:
            self.shift_size = int(window_size / 2)
            self.updown_mask = nn.Parameter(get_mask(window_size=self.window_size, shift_size=self.shift_size,
                                                     up_down=True, left_right=False), requires_grad=False)
            self.leftright_mask = nn.Parameter(get_mask(window_size=self.window_size, shift_size=self.shift_size,
                                                        up_down=False, left_right=True), requires_grad=False)

        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        self.head_dims = int(embedding_dims / num_heads)
        self.qkv = nn.Linear(embedding_dims, embedding_dims * 3)
        self.dropout = nn.Dropout(p=p_dropout)

        self.relative_pos_embeddings = relative_pos_embedding
        if self.relative_pos_embeddings:
            self.relative_indices = get_relative_distances(window_size=self.window_size) + window_size - 1
            self.pos_embeddings = nn.Parameter(torch.randn(2 * self.window_size - 1, 2 * self.window_size - 1))
        else:
            self.pos_embeddings = nn.Parameter(torch.randn(self.window_size ** 2, self.window_size ** 2))

    def forward(self, x):
        # cyclic shift
        if self.shifted:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))

        B, H, W, C = x.shape # (B, 56, 56, 96)
        x = self.qkv(x) # (B, 56, 56, 288)
        QKV = x.chunk(3, dim=-1)
        Q, K, V = map(
            lambda t: rearrange(t, 'b (wh ws1) (ww ws2) (n H) -> b n (wh ww) (ws1 ws2) H',
                                wh=int(H / self.window_size), ws1=self.window_size, ww=int(W / self.window_size), ws2=self.window_size,
                                n=self.num_heads, H=self.head_dims), QKV
        ) # (B, 8, 64, 49, 12)

        attention_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dims) # (B, 8, 64, 49, 49)
        if self.relative_pos_embeddings:
            attention_scores += self.pos_embeddings[self.relative_indices[:,:,0], self.relative_indices[:,:,1]]
        else:
            attention_scores += self.pos_embeddings

        if self.shifted:
            attention_scores[:,:,-W:] += self.updown_mask
            attention_scores[:,:,(W-1)::W] += self.leftright_mask

        attention_scores = torch.softmax(attention_scores, dim=-1) # (B, 8, 64, 49, 49)
        attention_scores = self.dropout(attention_scores) # (B, 8, 64, 49, 49)
        attention_outputs = torch.matmul(attention_scores, V) # (B, 8, 64, 49, 12)

        out = rearrange(attention_outputs, 'b n (wh ww) (ws1 ws2) H -> b (wh ws1) (ww ws2) (n H)',
                        wh=int(H / self.window_size), ws1=self.window_size, ww=int(W / self.window_size), ws2=self.window_size,
                        n=self.num_heads, H=self.head_dims) # (B, 56, 56, 96)

        if self.shifted:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(1,2))  # (B, 56, 56, 96)

        return out


class SwinEncoderBlock(nn.Module):
    def __init__(self, shifted, relative_pos_embedding, embedding_dims=96, num_heads=8, window_size=7, p_dropout=0.5):
        super(SwinEncoderBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embedding_dims=embedding_dims, num_heads=num_heads, window_size=window_size,
                                                p_dropout=p_dropout, shifted=shifted, relative_pos_embedding=relative_pos_embedding)
        self.norm = nn.LayerNorm(embedding_dims, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dims, 4 * embedding_dims),
            nn.GELU(),
            nn.Linear(4 * embedding_dims, embedding_dims)
        )

    def forward(self, x):
        x = x + self.norm(self.attention(x))
        x = x + self.norm(self.mlp(x))

        return x


class PatchMerging(nn.Module):
    def __init__(self, in_dims, out_dims, downscaling_factor):
        super(PatchMerging, self).__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_dims * downscaling_factor ** 2, out_dims)

    def forward(self, x):
        B, C, H, W = x.shape
        H_, W_ = int(H / self.downscaling_factor), int(W / self.downscaling_factor)

        x = self.patch_merge(x).view(B, -1, H_, W_).permute(0, 2, 3, 1)
        x = self.linear(x)

        return x


class SwinStageModule(nn.Module):
    def __init__(self, in_dims, downscaling_factor, num_block, embedding_dims=96, num_heads=8, window_size=7, p_dropout=0.5):
        super(SwinStageModule, self).__init__()
        self.patch_partition = PatchMerging(in_dims=in_dims, out_dims=embedding_dims, downscaling_factor=downscaling_factor)
        self.layers = nn.ModuleList([])
        for _ in range(int(num_block/2)):
            self.layers.append(nn.ModuleList([
                SwinEncoderBlock(shifted=False, relative_pos_embedding=True, embedding_dims=embedding_dims,
                                 num_heads=num_heads, window_size=window_size,p_dropout=p_dropout),
                SwinEncoderBlock(shifted=True, relative_pos_embedding=True, embedding_dims=embedding_dims,
                                 num_heads=num_heads, window_size=window_size, p_dropout=p_dropout)
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for swa, mswa in self.layers:
            x = swa(x)
            x = mswa(x)

        out = x.permute(0, 3, 1, 2)

        return out


class SwinTransformer(nn.Module):
    def __init__(self, in_channels, num_classes, downscaling_factors, num_heads, embedding_dims, num_blocks, p_dropout=0.5):
        super(SwinTransformer, self).__init__()
        self.stage1 = SwinStageModule(in_dims=in_channels, downscaling_factor=downscaling_factors[0], num_block=num_blocks[0],
                                      embedding_dims=embedding_dims*1, num_heads=num_heads*1, window_size=7, p_dropout=0.5)
        self.stage2 = SwinStageModule(in_dims=embedding_dims*1, downscaling_factor=downscaling_factors[1], num_block=num_blocks[1],
                                      embedding_dims=embedding_dims*2, num_heads=num_heads*2, window_size=7, p_dropout=0.5)
        self.stage3 = SwinStageModule(in_dims=embedding_dims*2, downscaling_factor=downscaling_factors[2], num_block=num_blocks[2],
                                      embedding_dims=embedding_dims*4, num_heads=num_heads*4, window_size=7, p_dropout=0.5)
        self.stage4 = SwinStageModule(in_dims=embedding_dims*4, downscaling_factor=downscaling_factors[3], num_block=num_blocks[3],
                                      embedding_dims=embedding_dims*8, num_heads=num_heads*8, window_size=7, p_dropout=0.5)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Linear(embedding_dims*8, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def swin_t(**kwargs):
    return SwinTransformer(downscaling_factors=[4,2,2,2], num_heads=3, embedding_dims=96, num_blocks=[2,2,6,2], p_dropout=0.5, **kwargs)


def swin_s(**kwargs):
    return SwinTransformer(downscaling_factors=[4,2,2,2], num_heads=3, embedding_dims=96, num_blocks=[2,2,18,2], p_dropout=0.5, **kwargs)


def swin_b(**kwargs):
    return SwinTransformer(downscaling_factors=[4,2,2,2], num_heads=4, embedding_dims=128, num_blocks=[2,2,18,2], p_dropout=0.5, **kwargs)


def swin_l(**kwargs):
    return SwinTransformer(downscaling_factors=[4,2,2,2], num_heads=6, embedding_dims=192, num_blocks=[2,2,18,2], p_dropout=0.5, **kwargs)


if __name__ == '__main__':
    img_size = 224
    in_channels = 3

    model = swin_t(in_channels=in_channels, num_classes=1000)

    input = torch.randn(8, in_channels, img_size, img_size)

    output = model(input)
    print(output.shape)