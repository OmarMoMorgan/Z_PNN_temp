import torch
import torch.nn as nn
import torch.nn.functional as F


class PNN(nn.Module):
    def __init__(self, in_channels, kernels, scope):
        super(PNN, self).__init__()

        # Network variables
        self.scope = scope

        # Network structure
        self.conv1 = nn.Conv2d(in_channels, 48, kernels[0])
        self.conv2 = nn.Conv2d(48, 32, kernels[1])
        self.conv3 = nn.Conv2d(32, in_channels - 1, kernels[2])

    def forward(self, inp):
        x = F.relu(self.conv1(inp))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x + inp[:, :-1, self.scope:-self.scope, self.scope:-self.scope]
        return x


class PanNet(nn.Module):
    def __init__(self, nbands, ratio):
        super(PanNet, self).__init__()

        bfilter_ms = torch.ones(nbands, 1, 5, 5)
        bfilter_ms = bfilter_ms / (bfilter_ms.shape[-2] * bfilter_ms.shape[-1])
        bfilter_pan = torch.ones(1, 1, 5, 5)
        bfilter_pan = bfilter_pan / (bfilter_pan.shape[-2] * bfilter_pan.shape[-1])

        self.dephtconv_ms = nn.Conv2d(in_channels=nbands, out_channels=nbands, padding=(2, 2),
                         kernel_size=bfilter_ms.shape, groups=nbands, bias=False, padding_mode='replicate')
        self.dephtconv_ms.weight.data = bfilter_ms
        self.dephtconv_ms.weight.requires_grad = False

        self.dephtconv_pan = nn.Conv2d(in_channels=1, out_channels=1, padding=(2, 2),
                                      kernel_size=bfilter_pan.shape, groups=1, bias=False, padding_mode='replicate')
        self.dephtconv_pan.weight.data = bfilter_pan
        self.dephtconv_pan.weight.requires_grad = False

        self.ratio = ratio
        self.Conv2d_transpose = nn.ConvTranspose2d(nbands, nbands, 8, 4, padding=(2, 2), bias=False)
        self.Conv = nn.Conv2d(nbands + 1, 32, 3, padding=(1, 1))
        self.Conv_1 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.Conv_2 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.Conv_3 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.Conv_4 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.Conv_5 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.Conv_6 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.Conv_7 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.Conv_8 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.Conv_9 = nn.Conv2d(32, nbands, 3, padding=(1, 1))

    def forward(self, inp):
        lms = inp[:, :-1, 2::self.ratio, 2::self.ratio]
        pan = torch.unsqueeze(inp[:, -1, :, :], dim=1)

        lms_hp = lms - self.dephtconv_ms(lms)
        pan_hp = pan - self.dephtconv_pan(pan)

        x = self.Conv2d_transpose(lms_hp)
        net_inp = torch.cat((x, pan_hp), dim=1)

        x1 = F.relu(self.Conv(net_inp))

        x2 = F.relu(self.Conv_1(x1))
        x3 = self.Conv_2(x2) + x1

        x4 = F.relu(self.Conv_3(x3))
        x5 = self.Conv_4(x4) + x3

        x6 = F.relu(self.Conv_5(x5))
        x7 = self.Conv_6(x6) + x5

        x8 = F.relu(self.Conv_7(x7))
        x9 = self.Conv_8(x8) + x7

        x10 = self.Conv_9(x9)

        x11 = inp[:, :-1, :, :] + x10

        return x11


class DRPNN(nn.Module):
    def __init__(self, in_channels):
        super(DRPNN, self).__init__()
        self.Conv_1 = nn.Conv2d(in_channels, 64, 7, padding=(3, 3))
        self.Conv_2 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_3 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_4 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_5 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_6 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_7 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_8 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_9 = nn.Conv2d(64, 64, 7, padding=(3, 3))
        self.Conv_10 = nn.Conv2d(64, in_channels, 7, padding=(3, 3))
        self.Conv_11 = nn.Conv2d(in_channels, in_channels - 1, 3, padding=(1, 1))

    def forward(self, x):
        x1 = F.relu(self.Conv_1(x))
        x2 = F.relu(self.Conv_2(x1))
        x3 = F.relu(self.Conv_3(x2))
        x4 = F.relu(self.Conv_4(x3))
        x5 = F.relu(self.Conv_5(x4))
        x6 = F.relu(self.Conv_6(x5))
        x7 = F.relu(self.Conv_7(x6))
        x8 = F.relu(self.Conv_8(x7))
        x9 = F.relu(self.Conv_9(x8))
        x10 = self.Conv_10(x9)
        x11 = self.Conv_11(F.relu(x10 + x))

        return x11



import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class CNNBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, groups=1
    ):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()  # SiLU <-> Swish

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        expand_ratio,
        reduction=4,  # squeeze excitation
        survival_prob=0.8,  # for stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride,
                padding,
                groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = (
            torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        )
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)
            

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class CNNBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, groups=1
    ):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()  # SiLU <-> Swish

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        expand_ratio,
        reduction=4,  # squeeze excitation
        survival_prob=0.8,  # for stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride,
                padding,
                groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = (
            torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        )
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, version, in_channels):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels,in_channels)

        self.Conv_10 = nn.Conv2d(64, in_channels, 7, padding=(3, 3))
        self.Conv_11 = nn.Conv2d(in_channels, in_channels - 1, 3, padding=(1, 1))
        # self.upsample = nn.Sequential(
        #     nn.Conv2d(320, 128, kernel_size=1),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(128, 64, kernel_size=1),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(64, 32, kernel_size=1),
        #     nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        #     nn.Conv2d(32, in_channels, kernel_size=1),
        # )

        # self.upsample = nn.Sequential(
        #     nn.Conv2d(320, 128, kernel_size=1),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(128, 64, kernel_size=1),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(64, out_channels, kernel_size=1),
        #     nn.Upsample(size=(16, 16), mode='bilinear', align_corners=True),
        # )



        # self.in_channels = in_channels
        # self.classifier = nn.Sequential(
            # nn.Dropout(dropout_rate),
            # nn.Linear(last_channels, num_classes),
        # )
        #self.convfinal = nn.Conv2d(in_channels, in_channels - 1, 3, padding=(1, 1))

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha**phi
        width_factor = beta**phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels,init_in_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(init_in_channels, channels, 3, stride=1, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=1, #stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                    )
                )
                in_channels = out_channels

        #features.append(
            #CNNBlock(in_channels, init_in_channels-1, kernel_size=1, stride=1, padding=(1,1))
            #self.convfinal
        #    nn.Conv2d(in_channels, last_channels, 3))
        features.append(
            #nn.Conv2d(in_channels, init_in_channels - 1, 3, padding=(1, 1))
            nn.Conv2d(in_channels, 64, 3, padding=(1, 1))
            #nn.Conv2d(in_channels, 320, 1)
        )

        return nn.Sequential(*features)

    def forward(self, x):
        #x = self.pool(self.features(x))
        # return self.classifier(x.view(x.shape[0],-1))
        x9 = self.features(x)
        x10 = self.Conv_10(x9)
        x11 = self.Conv_11(F.relu(x10 + x))
        #x = self.upsample(x)
        return x11

