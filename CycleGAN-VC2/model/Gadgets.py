"""
Useful modules
"""
import torch
import torch.nn as nn

class Conv2d_In_GLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2d_In_GLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.ins = nn.InstanceNorm2d(out_channels)
        self.conv_gate = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)
        self.ins_gate = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x1 = x
        first_x2 = self.ins(self.conv(x1))
        second_x2 = self.ins_gate(self.conv_gate(x1))
        return first_x2 * torch.sigmoid(second_x2)
