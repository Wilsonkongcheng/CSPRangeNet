# This file was modified from https://github.com/BobLiu20/YOLOv3_PyTorch
# It needed to be modified in order to accomodate for different strides in the

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import os

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class FirstResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, bn_d):
        super(FirstResidualBlock, self).__init__()
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel//2, 1, 1, 0, bias=False),
                                  nn.BatchNorm2d(outchannel//2, momentum=bn_d),
                                  Mish(),
                                  nn.Conv2d(outchannel//2, outchannel, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(outchannel, momentum= bn_d),
                                  Mish())

    def forward(self, x):
        return x + self.left(x)

#the first Block is different from the rest of blocks
class FirstCSPNetBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride, bn_d):    # [32,64]
        super(FirstCSPNetBlock, self).__init__()
    # down sampling
        self.front = nn.Sequential(nn.Conv2d(inchannel, outchannel, 3, [1, stride], 1, bias=False),
                                   nn.BatchNorm2d(outchannel, momentum=bn_d),
                                   Mish())
        self.right = nn.Sequential(nn.Conv2d(outchannel, outchannel, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(outchannel, momentum=bn_d),
                                   Mish())

        self.left = nn.Sequential(nn.Conv2d(outchannel, outchannel, 1, 1, 0, bias=False),
                                  nn.BatchNorm2d(outchannel, momentum=bn_d),
                                  Mish(),
                                  FirstResidualBlock(outchannel, outchannel, bn_d),
                                  nn.Conv2d(outchannel, outchannel, 1, 1, 0, bias=False),
                                  nn.BatchNorm2d(outchannel, momentum=bn_d),
                                  Mish())
        self.cat = nn.Sequential(nn.Conv2d(outchannel * 2, outchannel, 1, 1, 0, bias=False),
                                 nn.BatchNorm2d(outchannel, momentum=bn_d),
                                 Mish())

    def forward(self, x):
        x = self.front(x)
        left = self.left(x)
        right = self.right(x)
        out = torch.cat([left, right], dim=1)
        out = self.cat(out)
        return out

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, bn_d=0.1):   # 64  [64 128] ,
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
        self.mish1 = Mish()
        self.conv2 = nn.Conv2d(planes[0], planes[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[0], momentum=bn_d)
        self.mish2 = Mish()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.mish2(out)

        out += residual
        return out
class CSPNetBlock(nn.Module):
    def __init__(self, block, planes, blocks, bn_d=0.1):  # [64,128]
        super(CSPNetBlock, self).__init__()
        layers = []
        self.right = nn.Sequential(     # 128->64
            nn.Conv2d(planes[1], planes[0], kernel_size=1, stride=1, padding=0, bias=False),  # 64
            nn.BatchNorm2d(planes[0], momentum=bn_d),
            Mish())

        #  residual blocks
        inplanes = planes[0]   # 64
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i),
                           block(inplanes, planes, bn_d)))  # 64 [64,128]
        self.res = nn.Sequential(OrderedDict(layers))
        self.left = nn.Sequential(nn.Conv2d(planes[1], planes[0], kernel_size=1, stride=1, padding=0, bias=False),   # 128 64
                                  nn.BatchNorm2d(planes[0], momentum=bn_d),
                                  Mish(),
                                  self.res,
                                  nn.Conv2d(planes[0], planes[0], kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(planes[0], momentum=bn_d),
                                  Mish())

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        out = torch.cat([left, right], dim=1)    # 64->128
        return out


# ******************************************************************************

# number of layers per model
model_blocks = {
    21: [1, 1, 2, 2, 1],
    53: [1, 2, 8, 8, 4],
}


class Backbone(nn.Module):
    """
     Class for DarknetSeg. Subclasses PyTorch's own "nn" module
  """

    def __init__(self, params):
    # def __init__(self):
        super(Backbone, self).__init__()
        self.use_range = params["input_depth"]["range"]
        self.use_xyz = params["input_depth"]["xyz"]
        self.use_remission = params["input_depth"]["remission"]
        self.drop_prob = params["dropout"]
        self.bn_d = params["bn_d"]
        self.OS = params["OS"]
        self.layers = params["extra"]["layers"]
        print("Using CSPDarknetNet" + str(self.layers) + " Backbone")

        # input depth calc
        self.input_depth = 0
        self.input_idxs = []
        if self.use_range:
            self.input_depth += 1
            self.input_idxs.append(0)
        if self.use_xyz:
            self.input_depth += 3    # xyz
            self.input_idxs.extend([1, 2, 3])
            # self.input_depth += 2    # yaw_pitch
            # self.input_idxs.extend([1, 2])
        if self.use_remission:
            self.input_depth += 1
            self.input_idxs.append(4) #  xyz
            # self.input_idxs.append(3)   #  yaw_pitch

        print("Depth of backbone input = ", self.input_depth)

        # stride play
        self.strides = [2, 2, 2, 2, 2]
        # check current stride
        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Original OS: ", current_os)

        # make the new stride
        if self.OS > current_os:
            print("Can't do OS, ", self.OS,
                  " because it is bigger than original ", current_os)
        else:
            # redo strides according to needed stride
            for i, stride in enumerate(reversed(self.strides), 0):
                if int(current_os) != self.OS:
                    if stride == 2:
                        current_os /= 2
                        self.strides[-1 - i] = 1
                    if int(current_os) == self.OS:
                        break
            print("New OS: ", int(current_os))
            print("Strides: ", self.strides)

        # check that darknet exists
        assert self.layers in model_blocks.keys()

        # generate layers depending on darknet type
        self.blocks = model_blocks[self.layers]   # [ 1, 2 ,8 ,8 ,4]

        # input layer
        self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
        self.mish = Mish()      # 32

        # encoder

        self.enc1 = FirstCSPNetBlock(32, 64, stride=self.strides[0], bn_d=self.bn_d)
        self.enc2 = self._make_enc_layer(BasicBlock, [64, 128], self.blocks[1],
                                         stride=self.strides[1], bn_d=self.bn_d)
        self.enc3 = self._make_enc_layer(BasicBlock, [128, 256], self.blocks[2],
                                         stride=self.strides[2], bn_d=self.bn_d)
        self.enc4 = self._make_enc_layer(BasicBlock, [256, 512], self.blocks[3],
                                         stride=self.strides[3], bn_d=self.bn_d)
        self.enc5 = self._make_enc_layer(BasicBlock, [512, 1024], self.blocks[4],
                                         stride=self.strides[4], bn_d=self.bn_d)

        # for a bit of fun
        self.dropout = nn.Dropout2d(self.drop_prob)

        # last channels
        self.last_channels = 1024

    # make layer useful function   2 3 4 5
    def _make_enc_layer(self, block, planes, blocks, stride, bn_d):  # [64,128]
        layers = []

        #  downsample
        layers.append(("conv1", nn.Conv2d(planes[0], planes[1],   #   64 128
                                         kernel_size=3,
                                         stride=[1, stride], dilation=1,
                                         padding=1, bias=False)))
        layers.append(("bn1", nn.BatchNorm2d(planes[1], momentum=bn_d)))
        layers.append(("mish1", Mish()))
        #
        layers.append(("CSPNetBlock", CSPNetBlock(block, planes, blocks, bn_d)))

        layers.append(("conv2", nn.Conv2d(planes[1], planes[1], kernel_size=1, stride=1, padding=0, bias=False)))
        layers.append(("bn2", nn.BatchNorm2d(planes[1], momentum=bn_d)))
        layers.append(("mish2", Mish()))

        return nn.Sequential(OrderedDict(layers))



    def run_layer(self, x, layer, skips, os):
        y = layer(x)
        if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:   # W经过下采样，维度下降一半
            skips[os] = x.detach()      # 分离出当前分支，且不进行反向传播
            os *= 2
        x = y
        return x, skips, os

    def forward(self, x):
        # filter input
        x = x[:, self.input_idxs]

        # run cnn
        # store for skip connections
        skips = {}
        os = 1

        # first layer
        x, skips, os = self.run_layer(x, self.conv1, skips, os)
        x, skips, os = self.run_layer(x, self.bn1, skips, os)
        x, skips, os = self.run_layer(x, self.mish, skips, os)

        # all encoder blocks with intermediate dropouts
        x, skips, os = self.run_layer(x, self.enc1, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.enc2, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.enc3, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.enc4, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.enc5, skips, os)
        x, skips, os = self.run_layer(x, self.dropout, skips, os)

        return x, skips

    def get_last_depth(self):
        return self.last_channels

    def get_input_depth(self):
        return self.input_depth

if __name__ =="__main__":
    x = torch.rand(1,5,64,2048)
    model = Backbone()
    # # input
    # x = model.conv1(x) # [1, 32, 64, 2048]
    # x = model.bn1(x)
    # x = model.mish(x)  # [1, 32, 64, 2048]
    # #enc1 enc2  enc3
    # x = model.enc1(x)  # [1, 64, 64, 1024]
    # x = model.enc2(x)  # [1, 128, 64, 512]
    # x = model.enc3(x)  # [1, 256, 64, 256]
    # x = model.enc4(x)  # [1, 512, 64, 128]
    # x = model.enc5(x)  # [1, 1024, 64, 64]
    out, skips=  model.forward(x)
    print(out.shape)
    for k, v in skips.items():
        print(v.shape)