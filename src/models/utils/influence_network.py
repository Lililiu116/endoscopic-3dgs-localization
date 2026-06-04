# From https://github.com/yifita/deep_cage

import torch
import torch.nn as nn

class PointNetfeat(nn.Module):
    """
    From https://github.com/yifita/deep_cage
    """
    def __init__(self, dim=3, num_points=2500, global_feat=True, trans=False, bottleneck_size=512, activation="relu", normalization=None):
        super().__init__()
        self.conv1 = Conv1d(dim, 64, 1, activation=activation, normalization=normalization)
        # self.stn_embedding = STN(num_points = num_points, K=64)
        self.conv2 = Conv1d(64, 128, 1, activation=activation, normalization=normalization)
        self.conv3 = Conv1d(128, bottleneck_size, 1, activation=None, normalization=normalization)
        #self.mp1 = torch.nn.MaxPool1d(num_points)

        self.trans = trans
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = self.conv1(x)
        pointfeat = x
        x = self.conv2(x)
        x = self.conv3(x)
        x,_ = torch.max(x, dim=2)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(batchsize, -1, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x

class Linear(nn.Module):
    """
    1dconvolution with custom normalization and activation
    
    From https://github.com/yifita/deep_cage
    """

    def __init__(self, in_channels, out_channels, bias=True,
                 activation=None, normalization=None, momentum=0.01):
        super(Linear, self).__init__()
        self.activation = activation
        self.normalization = normalization
        bias = not normalization and bias
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

        if normalization is not None:
            if self.normalization == 'batch':
                self.norm = nn.BatchNorm1d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            elif self.normalization == 'instance':
                self.norm = nn.InstanceNorm1d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            else:
                raise ValueError(
                    "only \"batch/instance\" normalization permitted.")

        # activation
        if activation is not None:
            if self.activation == 'relu':
                self.act = nn.ReLU()
            elif self.activation == 'elu':
                self.act = nn.ELU(alpha=1.0)
            elif self.activation == 'lrelu':
                self.act = nn.LeakyReLU(0.1)
            elif self.activation == "tanh":
                self.act = nn.Tanh()
            else:
                raise ValueError("only \"relu/elu/lrelu/tanh\" implemented")

    def forward(self, x, epoch=None):
        x = self.linear(x)

        if self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)

        return x
    

class Conv1d(nn.Module):
    """
    1dconvolution with custom normalization and activation
    
    From https://github.com/yifita/deep_cage
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 activation=None, normalization=None, momentum=0.01, conv_params={}):
        super(Conv1d, self).__init__()
        self.activation = activation
        self.normalization = normalization
        bias = not normalization and bias
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias, **conv_params)

        if normalization is not None:
            if self.normalization == 'batch':
                self.norm = nn.BatchNorm1d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            elif self.normalization == 'instance':
                self.norm = nn.InstanceNorm1d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            else:
                raise ValueError(
                    "only \"batch/instance\" normalization permitted.")

        # activation
        if activation is not None:
            if self.activation == 'relu':
                self.act = nn.ReLU()
            elif self.activation == 'elu':
                self.act = nn.ELU(alpha=1.0)
            elif self.activation == 'lrelu':
                self.act = nn.LeakyReLU(0.1)
            elif self.activation == "tanh":
                self.act = nn.Tanh()
            else:
                raise ValueError("only \"relu/elu/lrelu/tanh\" implemented")

    def forward(self, x, epoch=None):
        x = self.conv(x)

        if self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)

        return x