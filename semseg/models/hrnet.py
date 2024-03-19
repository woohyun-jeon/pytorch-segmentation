import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BN_MOMENTUM = 0.1

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.downsample = downsample

        self.conv = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = F.relu(x)
        return x

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.stride = stride
        self.downsample = downsample

        self.conv = nn.Sequential(
            conv1x1(inplanes, planes),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = x + residual
        x = F.relu(x)
        return x

class HRModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HRModule,self).__init__()
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
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != self.num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], self.num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index], self.num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = self.num_channels[branch_index] * block.expansion
        for i in range(1, self.num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], self.num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self):
        branches = []
        for i in range(self.num_branches):
            branches.append(self._make_one_branch(i, self.blocks))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels

        fuse_layers = []




blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': BottleNeck
}

class HRNet(nn.Module):
    def __init__(self, in_channels, num_classes, cfg):
        super(HRNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # stem network
        self.conv_stem = nn.Sequential(
            conv3x3(self.in_channels, 64, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            conv3x3(64, 64, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        # stage 1
        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']] # bottleneck
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block=block, inplanes=64, planes=num_channels, num_blocks=num_blocks)
        stage1_out_channel = block.expansion * num_channels # 최종 채널

        # stage 2
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']] # basic
        num_channels = [num_channels[i]*block.expansion for i in range(len(num_channels))] # 최종 채널을 미리
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, inplanes=num_channels)

        # stage 3
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]  # basic
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]  # 최종 채널을 미리
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, inplanes=num_channels)

        # stage 4
        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]  # basic
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]  # 최종 채널을 미리
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, inplanes=num_channels)

        last_inp_channels = np.int(np.sum(pre_stage_channels))

        # last layer for segmentation
        self.last_layer = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(replace=True),
            nn.Conv2d(last_inp_channels, self.num_classes, kernel_size=3, stride=1, padding=1)
        )


    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_stage, num_channels_cur_stage):
        num_branches_pre = len(num_channels_pre_stage)
        num_branches_cur = len(num_channels_cur_stage)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_pre_stage[i] != num_channels_cur_stage[i]:
                    transition_layers.append(nn.Sequential(
                        conv3x3(num_channels_pre_stage[i], num_channels_cur_stage[i], bias=False),
                        nn.BatchNorm2d(num_channels_cur_stage[i], momentum=BN_MOMENTUM),
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
                        conv3x3(in_channels, out_channels, stride=2, group=1, bias=False),
                        nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
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
        x = self.conv_stem(x)
        x = self.layer1(x)







def _hrnet(arch, **kwargs):
    from .hrnet_config import MODEL_CONFIGS
    model = HRNet(MODEL_CONFIGS[arch], **kwargs)
    return model

def hrnet18(**kwargs):
    return _hrnet('hrnet18', **kwargs)

def hrnet32(**kwargs):
    return _hrnet('hrnet32', **kwargs)

def hrnet48(**kwargs):
    return _hrnet('hrnet48', **kwargs)

