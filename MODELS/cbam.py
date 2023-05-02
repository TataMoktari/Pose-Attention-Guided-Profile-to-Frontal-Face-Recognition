import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

        self.mlp_a = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, 896)
        )

        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        channel_att_sum_a = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
                channel_att_raw_a = self.mlp_a(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
                channel_att_raw_a = self.mlp_a(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

            if channel_att_sum_a is None:
                channel_att_sum_a = channel_att_raw_a
            else:
                channel_att_sum_a = channel_att_sum_a + channel_att_raw_a

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        # scale_a = F.sigmoid(channel_att_sum_a).unsqueeze(2).unsqueeze(3).expand_as(x)
        channel_matx = F.sigmoid(channel_att_sum_a)
        return x * scale, channel_matx


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        # kernel_size = 7
        # kernel_size2 = 3
        kernel_size1 = 1
        self.compress = ChannelPool()
        # self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        # self.spatial2 = BasicConv(2, 1, kernel_size2, stride=2, padding=0, relu=False)
        self.spatial1 = BasicConv(2, 1, kernel_size1, stride=1, padding=0, relu=False)
        # self.spatial2 = BasicConv(2, 1, kernel_size2, stride=2, padding=1, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        # x_out = self.spatial(x_compress)  # 7*7 attention_matrix
        # x_out2 = self.spatial2(x_compress)  # 3*3 attention_matrix
        x_out7 = self.spatial1(x_compress)  # 7*7 attention_matrix
        x_out8 = F.interpolate(x_out7, size=(8, 8), mode='bilinear')  # 8*8 attention_matrix using interpolation
        # scale = F.sigmoid(x_out)  # broadcasting
        # scale2 = F.sigmoid(x_out2)  # broadcasting
        scale8 = F.sigmoid(x_out8)  # broadcasting
        return scale8


class CBAM(nn.Module):
    # def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out, channel_matrix = self.ChannelGate(x)
        spatial_matrix = self.SpatialGate(x_out)
        return spatial_matrix, channel_matrix
