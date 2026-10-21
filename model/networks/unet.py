#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/23 22:26
    @Author : chairc
    @Site   : https://github.com/chairc
    Modified by Alexandre Stenger for GUDA: the network denoises a segmentation mask conditioned on the image and
    on a pseudo-label, which are concatenated to the noisy mask along the channel dimension. The depth, the channels,
    the attention resolutions and the number of attention heads are configurable.
"""
import torch
import torch.nn as nn

from model.networks.base import BaseNet
from model.modules.attention import SelfAttention
from model.modules.block import DownBlock, UpBlock
from model.modules.conv import DoubleConv


class UNetAttention(BaseNet):
    """
    UNet with self-attention

    The encoder has one level per entry of 'channel', the resolution being divided by 2 between levels, and the
    decoder mirrors it with skip connections. Self-attention is applied after the blocks working at a resolution
    listed in 'attention_resolutions'.
    """

    def __init__(self, in_channel=9, out_channel=3, channel=(128, 256, 384, 512), attention_resolutions=(16,),
                 num_heads=1, time_channel=256, image_size=64, device="cpu", act="silu"):
        """
        Initialize the UNet network
        :param in_channel: Input channel, noisy mask (3) + image (3) + pseudo-label (3)
        :param out_channel: Output channel, predicted noise on the mask
        :param channel: Channels of each level, from the input resolution to the lowest one
        :param attention_resolutions: Resolutions at which self-attention is applied
        :param num_heads: Number of attention heads
        :param time_channel: Time channel
        :param image_size: Adaptive image size
        :param device: Device type
        :param act: Activation function
        """
        super().__init__(in_channel, out_channel, list(channel), time_channel, image_size, device, act)
        if image_size % 2 ** (len(self.channel) - 1) != 0:
            raise ValueError(f"image_size {image_size} must be divisible by {2 ** (len(self.channel) - 1)}.")
        self.attention_resolutions = list(attention_resolutions)
        sizes = [image_size // 2 ** i for i in range(len(self.channel))]

        def attention(channels, size):
            if size in self.attention_resolutions:
                return SelfAttention(channels=channels, size=size, num_heads=num_heads, act=self.act)
            return nn.Identity()

        # Encoder, level 0 keeps the input resolution
        self.inc = DoubleConv(in_channels=self.in_channel, out_channels=self.channel[0], act=self.act)
        self.inc_sa = attention(self.channel[0], sizes[0])
        self.downs = nn.ModuleList()
        self.down_sas = nn.ModuleList()
        for i in range(1, len(self.channel)):
            self.downs.append(DownBlock(in_channels=self.channel[i - 1], out_channels=self.channel[i], act=self.act))
            self.down_sas.append(attention(self.channel[i], sizes[i]))

        # Bottleneck, at the lowest resolution
        self.bot = nn.Sequential(
            DoubleConv(in_channels=self.channel[-1], out_channels=self.channel[-1] * 2, act=self.act),
            DoubleConv(in_channels=self.channel[-1] * 2, out_channels=self.channel[-1] * 2, act=self.act),
            DoubleConv(in_channels=self.channel[-1] * 2, out_channels=self.channel[-1], act=self.act),
        )

        # Decoder, each block upsamples and merges the skip connection of the same level
        self.ups = nn.ModuleList()
        self.up_sas = nn.ModuleList()
        for i in reversed(range(1, len(self.channel))):
            self.ups.append(UpBlock(in_channels=self.channel[i] + self.channel[i - 1], out_channels=self.channel[i - 1],
                                    act=self.act))
            self.up_sas.append(attention(self.channel[i - 1], sizes[i - 1]))

        self.outc = nn.Conv2d(in_channels=self.channel[0], out_channels=self.out_channel, kernel_size=1)

    def forward(self, x, time):
        """
        Forward
        :param x: Input, concatenation of noisy mask, image and pseudo-label
        :param time: Time
        :return: output
        """
        time = time.unsqueeze(-1).type(torch.float)
        time = self.pos_encoding(time, self.time_channel)

        x = self.inc_sa(self.inc(x))
        skips = [x]
        for down, sa in zip(self.downs, self.down_sas):
            x = sa(down(x, time))
            skips.append(x)

        x = self.bot(x)

        for up, sa, skip in zip(self.ups, self.up_sas, reversed(skips[:-1])):
            x = sa(up(x, skip, time))
        return self.outc(x)
