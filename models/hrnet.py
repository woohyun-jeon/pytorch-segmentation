import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config.hrnet_config import MODEL_CONFIGS
from backbone.resnet import BasicBlock, Bottleneck


__all__ = ['HRNet', 'hrnet18', 'hrnet32', 'hrnet48']


class HRModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HRModule, self).__init__()
        self.num_branches = num_branches
        self.blocks = blocks
        self.num_blocks = num_blocks
        self.num_inchannels = num_inchannels
        self.num_channels = num_channels
        self.fuse_method = fuse_method
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches()
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, stride=1):
        block = self.blocks
        layers = []
        layers.append(block(self.num_inchannels[branch_index], self.num_channels[branch_index], stride))
        self.num_inchannels[branch_index] = self.num_channels[branch_index] * block.expansion
        for i in range(1, self.num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], self.num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self):
        branches = []
        for i in range(self.num_branches):
            branches.append(self._make_one_branch(branch_index=i))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels

        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(num_features=num_inchannels[i], momentum=0.01),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i-j-1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[i], kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_features=num_inchannels[i], momentum=0.01)
                            ))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[j], kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_features=num_inchannels[j], momentum=0.01),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=False
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HRNet(nn.Module):
    def __init__(self, in_channels, num_classes, cfg):
        super(HRNet, self).__init__()
        # stem network
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.01),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        # stage 1
        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block=block, inplanes=64, planes=num_channels, num_blocks=num_blocks)
        stage1_out_channel = block.expansion * num_channels

        # stage 2
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i]*block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_inchannels=num_channels)

        # stage 3
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_inchannels=num_channels)

        # stage 4
        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_inchannels=num_channels, multi_scale_output=True)

        last_inp_channels = int(np.sum(pre_stage_channels))

        # last layer for segmentation
        self.last_layer = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=last_inp_channels, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_inp_channels, num_classes, kernel_size=3, stride=1, padding=1)
        )

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(inplanes, planes))
            inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_stage, num_channels_cur_stage):
        num_branches_pre = len(num_channels_pre_stage)
        num_branches_cur = len(num_channels_cur_stage)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_pre_stage[i] != num_channels_cur_stage[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_stage[i], num_channels_cur_stage[i], kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(num_features=num_channels_cur_stage[i], momentum=0.01),
                        nn.ReLU(inplace=True))
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    in_channels = num_channels_pre_stage[-1]
                    out_channels = num_channels_cur_stage[i] if j == i-num_branches_pre else in_channels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(num_features=out_channels, momentum=0.01),
                        nn.ReLU(inplace=True))
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used at last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HRModule(num_branches,
                         block,
                         num_blocks,
                         num_inchannels,
                         num_channels,
                         fuse_method,
                         reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        out = self.conv_stem(x)
        out = self.layer1(out)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](out))
            else:
                x_list.append(out)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        out = self.stage4(x_list)

        # concat
        out1 = F.interpolate(out[1], size=out[0].shape[-2:], mode='bilinear', align_corners=False)
        out2 = F.interpolate(out[2], size=out[0].shape[-2:], mode='bilinear', align_corners=False)
        out3 = F.interpolate(out[3], size=out[0].shape[-2:], mode='bilinear', align_corners=False)

        out = torch.cat([out[0], out1, out2, out3], dim=1)

        out = self.last_layer(out)

        # upsample
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return out


def hrnet18(**kwargs):
    return HRNet(cfg=MODEL_CONFIGS['hrnet18'], **kwargs)


def hrnet32(**kwargs):
    return HRNet(cfg=MODEL_CONFIGS['hrnet32'], **kwargs)


def hrnet48(**kwargs):
    return HRNet(cfg=MODEL_CONFIGS['hrnet48'], **kwargs)


if __name__ == '__main__':
    img_size = 512
    model = hrnet48(in_channels=3, num_classes=1000)

    input = torch.randn(4, 3, img_size, img_size)
    output = model(input)

    print(output.shape)