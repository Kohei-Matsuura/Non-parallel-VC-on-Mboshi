"""
The discriminator of Cycle-GAN-VC2.
'h' (height) and 'w' (width) are replaced.
"""
import torch
import torch.nn as nn

import model.Gadgets as gad


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                                out_channels=128,
                                kernel_size=(3, 3),
                                stride=1,
                                padding=(1, 1))
        self.conv1_gate = nn.Conv2d(in_channels=1,
                                     out_channels=128,
                                     kernel_size=(3, 3),
                                     stride=1,
                                     padding=(1, 1))

        self.conv_in_glu1 = gad.Conv2d_In_GLU(in_channels=128,
                                               out_channels=256,
                                               kernel_size=(3, 3),
                                               stride=2,
                                               padding=(1, 1))
        self.conv_in_glu2 = gad.Conv2d_In_GLU(in_channels=256,
                                               out_channels=512,
                                               kernel_size=(3, 3),
                                               stride=2,
                                               padding=(1, 1))
        self.conv_in_glu3 = gad.Conv2d_In_GLU(in_channels=512,
                                               out_channels=1024,
                                               kernel_size=(3, 3),
                                               stride=2,
                                               padding=(1, 1))
        self.conv_in_glu4 = gad.Conv2d_In_GLU(in_channels=1024,
                                               out_channels=1024,
                                               kernel_size=(5, 1),
                                               stride=1,
                                               padding=(2, 0))

        self.conv2 = nn.Conv2d(in_channels=1024,
                                out_channels=1,
                                kernel_size=(3, 1),
                                stride=1,
                                padding=(1, 0))

    def forward(self, x):
        # Adding channel dim.
        x1 = x.unsqueeze(1)
        # Discriminating
        first_x2 = self.conv1(x1)
        second_x2 = self.conv1_gate(x1)
        x2 = first_x2 * torch.sigmoid(second_x2)
        x3 = self.conv_in_glu1(x2)
        x4 = self.conv_in_glu2(x3)
        x5 = self.conv_in_glu3(x4)
        x6 = self.conv_in_glu4(x5)
        x7 = self.conv2(x6)
        # Removing channel_dim
        x7 = x7.squeeze(1)
        return x7


# class MultiDiscriminator(Discriminator):
#     r"""This Discriminator can only accept 128-frame input.
#     It outputs one tensor and one scaler to realize multitask
#     learning between PatchGAN & normal GAN.
#     Full and patched images are considered to decide whether real/fake.
#     """
#     def __init__(self):
#         super(MultiDiscriminator, self).__init__()
#         self.conv_in_glu5 = gad.Conv2d_In_GLU(in_channels=512,
#                                                out_channels=1024,
#                                                kernel_size=(3, 6),
#                                                stride=(1, 2),
#                                                padding=(1, 2))
#         self.lin = nn.Linear(1024, 1)
#
#     def forward(self, x):
#         x1 = x
#         first_x2 = self.conv1(x1)
#         second_x2 = self.conv1_gate(x1)
#         x2 = first_x2 * torch.sigmoid(second_x2)
#         x3 = self.conv_in_glu1(x2)
#         x4 = self.conv_in_glu2(x3)
#         x5 = self.conv_in_glu3(x4)
#         x6 = self.conv_in_glu4(x5)
#         patch_x = self.conv2(x6)
#
#         x7 = self.conv_in_glu5(x4)
#         print(x7.shape)
#         return patch_x, x7
