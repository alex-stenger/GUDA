#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/12/5 10:22
    @Author : chairc
    @Site   : https://github.com/chairc
    Modified by Alexandre Stenger for GUDA: removed BaseConv.
"""
import logging

import torch.nn as nn
import torch.nn.functional as F

from model.modules.activation import get_activation_function

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """
    Double convolution
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, act="silu"):
        """
        Initialize the double convolution block
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param mid_channels: Middle channels
        :param residual: Whether residual
        :param act: Activation function
        """
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.act = act
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            get_activation_function(name=self.act),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )

    def forward(self, x):
        """
        DoubleConv forward
        :param x: Input
        :return: Residual or non-residual results
        """
        if self.residual:
            out = x + self.double_conv(x)
            if self.act == "relu":
                return F.relu(out)
            elif self.act == "relu6":
                return F.relu6(out)
            elif self.act == "silu":
                return F.silu(out)
            elif self.act == "lrelu":
                return F.leaky_relu(out)
            elif self.act == "gelu":
                return F.gelu(out)
            else:
                logger.warning(msg=f"Unsupported activation function type: {self.act}")
                return F.silu(out)
        else:
            return self.double_conv(x)
