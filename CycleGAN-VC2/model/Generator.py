"""
The generator of Cycle-GAN-VC2.
'h' (height) and 'w' (width) are replaced.
"""

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=256,
                                out_channels=512,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.in1 = nn.InstanceNorm1d(512)
        self.conv1_gate = nn.Conv1d(in_channels=256,
                                out_channels=512,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.in1_gate = nn.InstanceNorm1d(512)

        self.conv2 = nn.Conv1d(in_channels=512,
                                out_channels=256,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.in2 = nn.InstanceNorm1d(256)

    def forward(self, x):
        x1 = x
        first_x2 = self.in1(self.conv1(x1))
        second_x2 = self.in1_gate(self.conv1_gate(x1))
        x2 = first_x2 * torch.sigmoid(second_x2)
        x3 = self.conv2(x2)
        x4 = self.in2(x3)
        return x + x3


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # For Preprocessing ----------------------------
        self.conv1 = nn.Conv2d(in_channels=1,
                                out_channels=128,
                                kernel_size=(15, 5),
                                stride=1,
                                padding=(7, 2))
        self.conv1_gate = nn.Conv2d(in_channels=1,
                                     out_channels=128,
                                     kernel_size=(15, 5),
                                     stride=1,
                                     padding=(7, 2))

        # For Down Sampling ---------------------------
        self.conv2 = nn.Conv2d(in_channels=128,
                                out_channels=256,
                                kernel_size=(5, 5),
                                stride=2,
                                padding=(2, 2))
        self.in2 = nn.InstanceNorm2d(256)
        self.conv2_gate = nn.Conv2d(in_channels=128,
                                     out_channels=256,
                                     kernel_size=(5, 5),
                                     stride=2,
                                     padding=(2, 2))
        self.in2_gate = nn.InstanceNorm2d(256)

        self.conv3 = nn.Conv2d(in_channels=256,
                                out_channels=512,
                                kernel_size=(5, 5),
                                stride=2,
                                padding=(2, 2))
        self.in3 = nn.InstanceNorm2d(512)
        self.conv3_gate = nn.Conv2d(in_channels=256,
                                     out_channels=512,
                                     kernel_size=(5, 5),
                                     stride=2,
                                     padding=(2, 2))
        self.in3_gate = nn.InstanceNorm2d(512)

        # For Reshaping from 2d into 1d ---------------
        self.conv4 = nn.Conv1d(in_channels=5120,
                                out_channels=256,
                                kernel_size=1,
                                stride=1)
        self.in4 = nn.InstanceNorm1d(512)

        # For Main Conversion Part --------------------
        self.res1 = ResidualBlock()
        self.res2 = ResidualBlock()
        self.res3 = ResidualBlock()
        self.res4 = ResidualBlock()
        self.res5 = ResidualBlock()
        self.res6 = ResidualBlock()

        # For Reshaping from 1d into 2d ---------------
        self.conv5 = nn.Conv1d(in_channels=256,
                                out_channels=5120,
                                kernel_size=1,
                                stride=1)
        self.in5 = nn.InstanceNorm1d(5120)

        # For Up Sampling -----------------------------
        self.conv6 = nn.Conv2d(in_channels=512,
                                out_channels=1024,
                                kernel_size=(5, 5),
                                stride=1,
                                padding=(2, 2))
        self.in6 = nn.InstanceNorm2d(1024)
        self.ps1 = nn.PixelShuffle(2)
        self.conv6_gate = nn.Conv2d(in_channels=512,
                                     out_channels=1024,
                                     kernel_size=(5, 5),
                                     stride=1,
                                     padding=(2, 2))
        self.in6_gate = nn.InstanceNorm2d(1024)
        self.ps1_gate = nn.PixelShuffle(2)

        self.conv7 = nn.Conv2d(in_channels=256,
                                out_channels=512,
                                kernel_size=(5, 5),
                                stride=1,
                                padding=(2, 2))
        self.in7 = nn.InstanceNorm2d(512)
        self.ps2 = nn.PixelShuffle(2)
        self.conv7_gate = nn.Conv2d(in_channels=256,
                                     out_channels=512,
                                     kernel_size=(5, 5),
                                     stride=1,
                                     padding=(2, 2))
        self.in7_gate = nn.InstanceNorm2d(512)
        self.ps2_gate = nn.PixelShuffle(2)

        # For Postprocessing -------------------------
        self.conv8 = nn.Conv2d(in_channels=128,
                                out_channels=1,
                                kernel_size=(15, 5),
                                stride=1,
                                padding=(7, 2))

    def forward(self, x):
        # x: (B, T, F)
        B, T, F = x.shape

        # Preprocessing ------------------------------
        # Adding channel dim.
        x1 = x.unsqueeze(1)
        first_x2 = self.conv1(x1)
        second_x2 = self.conv1_gate(x1)
        x2 = first_x2 * torch.sigmoid(second_x2)

        # Down Sampling ------------------------------
        first_x3 = self.in2(self.conv2(x2))
        second_x3 = self.in2_gate(self.conv2_gate(x2))
        x3 = first_x3 * torch.sigmoid(second_x3)

        first_x4 = self.in3(self.conv3(x3))
        second_x4 = self.in3_gate(self.conv3_gate(x3))
        x4 = first_x4 * torch.sigmoid(second_x4)

        # Reshaping from 2d into 1d ------------------
        x4 = x4.view(B, -1, int(T/4))
        x5 = self.in4(self.conv4(x4))

        # Main Conversion Part -----------------------
        x6 = self.res1(x5)
        x7 = self.res2(x6)
        x8 = self.res3(x7)
        x9 = self.res4(x8)
        x10 = self.res5(x9)
        x11 = self.res6(x10)

        # Reshaping from 1d into 2d ------------------
        x12 = self.in5(self.conv5(x11))
        x12 = x12.view(B, -1, int(T/4), int(F/4))

        # Up Sampling --------------------------------
        x13_first = self.in6(self.ps1(self.conv6(x12)))
        x13_second = self.in6_gate(self.ps1_gate(self.conv6_gate(x12)))
        x13 = x13_first * torch.sigmoid(x13_second)

        x14_first = self.in7(self.ps2(self.conv7(x13)))
        x14_second = self.in7_gate(self.ps2_gate(self.conv7_gate(x13)))
        x14 = x14_first * torch.sigmoid(x14_second)

        # Post Processing ----------------------------
        x15 = self.conv8(x14)
        # Removing channel_dim
        x15 = x15.squeeze(1)

        return x15
