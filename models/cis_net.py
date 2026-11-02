"""
CIS-Net (Contextual Inpainting & Synthesis Network)
Deep Gated Convolutional Network for Photorealistic Watermark Texture Inpainting.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.neural_runtime import (
    Conv2D, GatedConv2D, ResidualDenseBlock, CBAMBlock,
    leaky_relu, sigmoid, relu
)

class ContextualInpaintingNet:
    """
    Gated Convolutional Inpainting Network with Contextual Attention.
    Takes 4-channel input (RGB Image + 1-channel Mask) and outputs reconstructed RGB.
    """
    def __init__(self, in_channels=4, base_channels=24):
        self.in_channels = in_channels
        self.base = base_channels

        # Gated Convolutions
        self.gate1 = GatedConv2D(in_channels, base_channels, kernel_size=5, stride=1, padding=2)
        self.gate2 = GatedConv2D(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1)

        # Dilated contextual blocks (capturing wider context surrounding the watermark)
        self.dilated1 = GatedConv2D(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1)
        self.dilated2 = GatedConv2D(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1)

        # Reconstruction & Texture Synthesis
        self.rdb = ResidualDenseBlock(base_channels * 2, growth_rate=16)
        self.gate3 = GatedConv2D(base_channels * 2, base_channels, kernel_size=3, stride=1, padding=1)
        self.out_conv = Conv2D(base_channels, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, img, mask):
        """
        Args:
            img: (N, 3, H, W) in [0, 1]
            mask: (N, 1, H, W) where 1.0 indicates pixels to inpaint
        Returns:
            inpainted: (N, 3, H, W) reconstructed image
        """
        self.img = img
        self.mask = mask
        self.masked_img = img * (1.0 - mask)
        self.x = np.concatenate([self.masked_img, mask], axis=1) # (N, 4, H, W)

        # Forward through gated layers
        self.h1 = self.gate1.forward(self.x)
        self.h2 = self.gate2.forward(self.h1)

        # Contextual dilation
        self.d1 = self.dilated1.forward(self.h2)
        self.d2 = self.dilated2.forward(self.d1)

        # Residual refinement
        self.r = self.rdb.forward(self.d2)

        # Synthesis
        self.gate3_in = self.r + self.h2 # skip connection
        self.h3 = self.gate3.forward(self.gate3_in)
        self.out_pre = self.out_conv.forward(self.h3)
        self.residual_rgb = sigmoid(self.out_pre)

        # Alpha-blend reconstructed core with original context
        self.inpainted = img * (1.0 - mask) + self.residual_rgb * mask
        return self.inpainted

    def backward(self, d_inpainted):
        """
        Backpropagation through Contextual Inpainting Network.
        """
        d_res = d_inpainted * self.mask
        d_out_pre = d_res * self.residual_rgb * (1.0 - self.residual_rgb)
        d_h3 = self.out_conv.backward(d_out_pre)

        d_gate3_in = self.gate3.backward(d_h3)
        d_r = d_gate3_in
        d_h2_skip = d_gate3_in

        d_d2 = self.rdb.backward(d_r)
        d_d1 = self.dilated2.backward(d_d2)
        d_h2 = self.dilated1.backward(d_d1) + d_h2_skip

        d_h1 = self.gate2.backward(d_h2)
        d_x = self.gate1.backward(d_h1)
        return d_x

    def get_trainable_layers(self):
        """Collect all trainable Conv2D layers."""
        return [
            self.gate1.conv_f, self.gate1.conv_g,
            self.gate2.conv_f, self.gate2.conv_g,
            self.dilated1.conv_f, self.dilated1.conv_g,
            self.dilated2.conv_f, self.dilated2.conv_g,
            self.rdb.conv1, self.rdb.conv2, self.rdb.conv3,
            self.gate3.conv_f, self.gate3.conv_g,
            self.out_conv
        ]

    def get_weights(self):
        """Export all trainable weights as a state dictionary."""
        state = {}
        for attr, val in self.__dict__.items():
            if isinstance(val, Conv2D):
                state[f"{attr}.W"] = val.W
                if val.use_bias:
                    state[f"{attr}.b"] = val.b
            elif isinstance(val, GatedConv2D):
                state[f"{attr}.conv_f.W"] = val.conv_f.W
                if val.conv_f.use_bias:
                    state[f"{attr}.conv_f.b"] = val.conv_f.b
                state[f"{attr}.conv_g.W"] = val.conv_g.W
                if val.conv_g.use_bias:
                    state[f"{attr}.conv_g.b"] = val.conv_g.b
            elif isinstance(val, ResidualDenseBlock):
                state[f"{attr}.conv1.W"] = val.conv1.W
                if val.conv1.use_bias:
                    state[f"{attr}.conv1.b"] = val.conv1.b
                state[f"{attr}.conv2.W"] = val.conv2.W
                if val.conv2.use_bias:
                    state[f"{attr}.conv2.b"] = val.conv2.b
                state[f"{attr}.conv3.W"] = val.conv3.W
                if val.conv3.use_bias:
                    state[f"{attr}.conv3.b"] = val.conv3.b
        return state

    def load_weights(self, state):
        """Load state dictionary."""
        for k, v in state.items():
            parts = k.split('.')
            mod = getattr(self, parts[0], None)
            if mod:
                if len(parts) == 2:
                    setattr(mod, parts[1], v.copy())
                elif len(parts) == 3:
                    sub = getattr(mod, parts[1], None)
                    if sub:
                        setattr(sub, parts[2], v.copy())
