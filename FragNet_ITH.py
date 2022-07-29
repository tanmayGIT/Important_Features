
# Sheng He, Lambert Schomaker,  FragNet: Writer Identification Using Deep Fragment Networks,
# IEEE Transactions on Information Forensics and Security ( Volume: 15), Pages: 3013-3022
# @Arixv: https://arxiv.org/pdf/2003.07212.pdf

# *************************


import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGnet(nn.Module):

    def __init__(self, input_channel):
        super().__init__()

        layers = [64, 128, 256, 512]

        self.conv1 = self._conv(input_channel, layers[0])
        self.maxp1 = nn.MaxPool2d(2, stride=2)
        # reduce the spatial size of the feature maps and obtain translation invariance
        self.conv2 = self._conv(layers[0], layers[1])
        self.maxp2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = self._conv(layers[1], layers[2])
        self.maxp3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = self._conv(layers[2], layers[3])
        self.maxp4 = nn.MaxPool2d(2, stride=2)

    def _conv(self, inplance, outplance, nlayers=2):
        conv = []
        # Conv2d(input channel (image), num_convolutional features, square_kernel size,
        #         stride,padding, bias, .. )
        for n in range(nlayers):
            conv.append(nn.Conv2d(inplance, outplance, kernel_size=3, stride=1, padding=1, bias=False))
            conv.append(nn.BatchNorm2d(outplance))
            conv.append(nn.ReLU(inplace=True))
            inplance = outplance  # build upon previous

        conv = nn.Sequential(*conv)

        return conv

    def forward(self, x):
        xlist = [x]  # the first element in the x list
        x = self.conv1(x)
        xlist.append(x)  # put as the second element in the list
        x = self.maxp1(x)
        x = self.conv2(x)
        xlist.append(x)  # put as the third element in the list
        x = self.maxp2(x)
        x = self.conv3(x)
        xlist.append(x)  # put as the fourth element in the list
        x = self.maxp3(x)
        x = self.conv4(x)
        xlist.append(x)  # put as the fifth element in the list
        return xlist


class Fragnet(nn.Module):
    def __init__(self, inplace, num_output_features):
        super().__init__()

        self.net = VGGnet(inplace)

        layers = [64, 128, 256, 512, 512]

        self.conv0 = self._conv(inplace, layers[0])
        self.conv1 = self._conv(layers[0] * 2, layers[1])
        self.maxp1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = self._conv(layers[1] * 2, layers[2])
        self.maxp2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = self._conv(layers[2] * 2, layers[3])
        self.maxp3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = self._conv(layers[3] * 2, layers[4])
        self.maxp4 = nn.MaxPool2d(2, stride=2)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.last_layer_1 = nn.Linear(512, 256)  # in_features, out_features, bias
        self.last_layer_2 = nn.Linear(256, 128)

        self.bn1 = nn.BatchNorm1d(256, eps=2e-1)
        self.bn2 = nn.BatchNorm1d(128, eps=2e-1)

        self.classifier = nn.Linear(128, num_output_features)

    def _conv(self, inplance, outplance, nlayers=2):
        conv = []
        for n in range(nlayers):
            conv.append(nn.Conv2d(inplance, outplance, kernel_size=3, stride=1, padding=1, bias=False))
            conv.append(nn.BatchNorm2d(outplance))
            conv.append(nn.ReLU(inplace=True))
            inplance = outplance

        conv = nn.Sequential(*conv)
        return conv

    def forward(self, x):
        xlist = self.net(x)

        step = 16

        # input image; so it is the first element of the list
        reslist = []
        for n in range(0, 65, step):
            xpatch = xlist[0][:, :, :, n:n + 64]  # cropping into the blocks of 64 cols; then taking 16 pixels jumps
            r = self.conv0(xpatch)
            reslist.append(r)

        # 0-layer
        idx = 0
        res1list = []
        for n in range(0, 65, step):
            xpatch = xlist[1][:, :, :, n:n + 64]  # took the second element of xlist and divide it into the blocks
            xpatch = torch.cat([xpatch, reslist[idx]], 1)
            idx += 1
            r = self.conv1(xpatch)
            r = self.maxp1(r)
            res1list.append(r)

        # 1-layer
        idx = 0
        res2list = []
        step = 8
        for n in range(0, 33, step):
            xpatch = xlist[2][:, :, :, n:n + 32]  # took the third element of xlist and divide it into the blocks
            xpatch = torch.cat([xpatch, res1list[idx]], 1)
            idx += 1
            r = self.conv2(xpatch)
            r = self.maxp2(r)
            res2list.append(r)

        # 2-layer

        idx = 0
        res3list = []
        step = 4
        for n in range(0, 17, step):
            xpatch = xlist[3][:, :, :, n:n + 16]  # took the fourth element of xlist and divide it into the blocks
            xpatch = torch.cat([xpatch, res2list[idx]], 1)
            idx += 1
            r = self.conv3(xpatch)
            r = self.maxp3(r)
            res3list.append(r)

        # 3-layer
        idx = 0
        step = 2
        logits_list_probs = []
        logits_list_no_softmax = []

        for n in range(0, 9, step):
            xpatch = xlist[4][:, :, :, n:n + 8]
            xpatch = torch.cat([xpatch, res3list[idx]], 1)
            idx += 1
            r = self.conv4(xpatch)
            r = self.maxp4(r)
            r = torch.flatten(self.avg(r), 1)  # global avg pooling, and global feature vector

            r = self.last_layer_1(r)
            r = F.relu(r)
            r = self.bn1(r)

            r = self.last_layer_2(r)
            r = F.relu(r)
            r = self.bn2(r)

            c = self.classifier(r)  # the final classification layer (without softmax)
            c_softmax = F.softmax(c, dim=1)

            logits_list_probs.append(c_softmax)  # store the posterior probabilities  for each patch
            logits_list_no_softmax.append(c)

        mean_1 = torch.mean(torch.stack(logits_list_probs), 0)
        mean_no_soft_max = torch.mean(torch.stack(logits_list_no_softmax), 0)

        return mean_1, mean_no_soft_max
