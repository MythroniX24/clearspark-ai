"""
WAL-Net (Watermark Attention Localization Network)
Specialized Deep Neural Network for Gemini / Imagen 3 Sparkle Detection.
Outputs pixel-level probability mask and normalized bounding box regression.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.neural_runtime import (
    Conv2D, ResidualDenseBlock, CBAMBlock, Upsample2D,
    leaky_relu, sigmoid, relu
)

class SparkleAttentionUNet:
    """
    Attention U-Net with Residual Dense Blocks & CBAM for high-precision
    sub-pixel watermark localization and probability mask generation.
    """
    def __init__(self, in_channels=3, base_channels=24):
        self.in_channels = in_channels
        self.base = base_channels

        # Encoder
        self.stem = Conv2D(in_channels, base_channels, kernel_size=3, padding=1)
        self.rdb0 = ResidualDenseBlock(base_channels, growth_rate=12)

        self.down1 = Conv2D(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.rdb1 = ResidualDenseBlock(base_channels * 2, growth_rate=16)

        self.down2 = Conv2D(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.bottleneck = CBAMBlock(base_channels * 4)

        # Decoder
        self.up1 = Upsample2D(scale_factor=2)
        self.conv_up1 = Conv2D(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        self.fuse1 = Conv2D(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)

        self.up2 = Upsample2D(scale_factor=2)
        self.conv_up2 = Conv2D(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.fuse2 = Conv2D(base_channels * 2, base_channels, kernel_size=3, padding=1)

        # Multi-task Heads
        self.mask_head = Conv2D(base_channels, 1, kernel_size=3, padding=1)
        self.conf_conv = Conv2D(base_channels * 4, 16, kernel_size=1, padding=0)
        self.conf_fc = Conv2D(16, 5, kernel_size=1, padding=0) # [conf, x, y, w, h]

        # Intermediate cache for backprop
        self.x = None
        self.s0_pre = None
        self.s0 = None
        self.s0_rdb = None
        self.d1_pre = None
        self.d1 = None
        self.d1_rdb = None
        self.d2_pre = None
        self.d2 = None
        self.b = None
        self.gap = None
        self.conf_feat_pre = None
        self.conf_feat = None
        self.box_out = None
        self.conf = None
        self.bbox = None
        self.u1_raw = None
        self.u1 = None
        self.cat1 = None
        self.f1_pre = None
        self.f1 = None
        self.u2_raw = None
        self.u2 = None
        self.cat2 = None
        self.f2_pre = None
        self.f2 = None
        self.mask_pre = None
        self.mask = None

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor of shape (N, C, H, W) normalized to [0, 1].
        Returns:
            dict containing:
                'mask': Soft probability mask (N, 1, H, W)
                'bbox': Bounding box prediction (N, 4) -> [cx, cy, w, h]
                'confidence': Detection confidence score (N, 1) in [0, 1]
        """
        self.x = x
        N, C, H, W = x.shape

        # Encoder Stem
        self.s0_pre = self.stem.forward(x)
        self.s0 = leaky_relu(self.s0_pre)
        self.s0_rdb = self.rdb0.forward(self.s0)

        # Down 1 (H/2, W/2)
        self.d1_pre = self.down1.forward(self.s0_rdb)
        self.d1 = leaky_relu(self.d1_pre)
        self.d1_rdb = self.rdb1.forward(self.d1)

        # Down 2 (H/4, W/4)
        self.d2_pre = self.down2.forward(self.d1_rdb)
        self.d2 = leaky_relu(self.d2_pre)
        self.b = self.bottleneck.forward(self.d2)

        # Confidence & Bounding Box regression from bottleneck
        self.gap = np.mean(self.b, axis=(2, 3), keepdims=True)
        self.conf_feat_pre = self.conf_conv.forward(self.gap)
        self.conf_feat = relu(self.conf_feat_pre)
        self.box_out = self.conf_fc.forward(self.conf_feat).reshape(N, 5)
        self.conf = sigmoid(self.box_out[:, 0:1])
        self.bbox = sigmoid(self.box_out[:, 1:5]) # normalized to [0, 1]

        # Decoder Up 1
        self.u1_raw = self.up1.forward(self.b)
        self.u1 = self.conv_up1.forward(self.u1_raw)
        # Match dimensions if odd padding
        if self.u1.shape[2:] != self.d1_rdb.shape[2:]:
            self.u1 = self.u1[:, :, :self.d1_rdb.shape[2], :self.d1_rdb.shape[3]]
        self.cat1 = np.concatenate([self.u1, self.d1_rdb], axis=1)
        self.f1_pre = self.fuse1.forward(self.cat1)
        self.f1 = leaky_relu(self.f1_pre)

        # Decoder Up 2
        self.u2_raw = self.up2.forward(self.f1)
        self.u2 = self.conv_up2.forward(self.u2_raw)
        if self.u2.shape[2:] != self.s0_rdb.shape[2:]:
            self.u2 = self.u2[:, :, :self.s0_rdb.shape[2], :self.s0_rdb.shape[3]]
        self.cat2 = np.concatenate([self.u2, self.s0_rdb], axis=1)
        self.f2_pre = self.fuse2.forward(self.cat2)
        self.f2 = leaky_relu(self.f2_pre)

        # Pixel Heatmap Mask
        self.mask_pre = self.mask_head.forward(self.f2)
        self.mask = sigmoid(self.mask_pre)

        return {
            'mask': self.mask,
            'bbox': self.bbox,
            'confidence': self.conf
        }

    def backward(self, d_mask, d_bbox=None, d_conf=None):
        """
        Backpropagation pass through Attention U-Net.
        Calculates dW, db across all internal convolutional layers.
        """
        N, _, H, W = self.mask.shape

        # 1. Mask Head
        # BCE + Sigmoid derivative: d_mask_pre = d_mask * mask * (1 - mask)
        d_mask_pre = d_mask * self.mask * (1.0 - self.mask)
        d_f2 = self.mask_head.backward(d_mask_pre)

        # 2. Decoder Up 2
        d_f2_pre = np.where(self.f2_pre > 0, d_f2, d_f2 * 0.2)
        d_cat2 = self.fuse2.backward(d_f2_pre)
        u2_c = self.u2.shape[1]
        d_u2 = d_cat2[:, :u2_c, :, :]
        d_s0_rdb = d_cat2[:, u2_c:, :, :]

        if d_u2.shape[2:] != self.u2_raw.shape[2:]:
            pad_h = self.u2_raw.shape[2] - d_u2.shape[2]
            pad_w = self.u2_raw.shape[3] - d_u2.shape[3]
            d_u2 = np.pad(d_u2, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))

        d_u2_raw = self.conv_up2.backward(d_u2)
        d_f1 = self.up2.backward(d_u2_raw)

        # 3. Decoder Up 1
        d_f1_pre = np.where(self.f1_pre > 0, d_f1, d_f1 * 0.2)
        d_cat1 = self.fuse1.backward(d_f1_pre)
        u1_c = self.u1.shape[1]
        d_u1 = d_cat1[:, :u1_c, :, :]
        d_d1_rdb = d_cat1[:, u1_c:, :, :]

        if d_u1.shape[2:] != self.u1_raw.shape[2:]:
            pad_h = self.u1_raw.shape[2] - d_u1.shape[2]
            pad_w = self.u1_raw.shape[3] - d_u1.shape[3]
            d_u1 = np.pad(d_u1, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))

        d_u1_raw = self.conv_up1.backward(d_u1)
        d_b_up = self.up1.backward(d_u1_raw)

        # 4. Box & Conf Heads
        if d_bbox is not None and d_conf is not None:
            d_conf_pre = (d_conf * self.conf * (1.0 - self.conf)).reshape(N, 1)
            d_bbox_pre = (d_bbox * self.bbox * (1.0 - self.bbox)).reshape(N, 4)
            d_box_out = np.concatenate([d_conf_pre, d_bbox_pre], axis=1).reshape(N, 5, 1, 1)
            d_conf_feat = self.conf_fc.backward(d_box_out)
            d_conf_feat_pre = np.where(self.conf_feat_pre > 0, d_conf_feat, 0.0)
            d_gap = self.conf_conv.backward(d_conf_feat_pre)
            bh, bw = self.b.shape[2], self.b.shape[3]
            d_b_gap = np.repeat(np.repeat(d_gap / float(bh * bw), bh, axis=2), bw, axis=3)
        else:
            d_b_gap = 0.0

        # 5. Bottleneck CBAM
        d_b = d_b_up + d_b_gap
        d_d2 = self.bottleneck.backward(d_b)

        # 6. Encoder Down 2
        d_d2_pre = np.where(self.d2_pre > 0, d_d2, d_d2 * 0.2)
        d_d1_rdb_from_down2 = self.down2.backward(d_d2_pre)
        d_d1_rdb_total = d_d1_rdb + d_d1_rdb_from_down2
        d_d1 = self.rdb1.backward(d_d1_rdb_total)

        # 7. Encoder Down 1
        d_d1_pre = np.where(self.d1_pre > 0, d_d1, d_d1 * 0.2)
        d_s0_rdb_from_down1 = self.down1.backward(d_d1_pre)
        d_s0_rdb_total = d_s0_rdb + d_s0_rdb_from_down1
        d_s0 = self.rdb0.backward(d_s0_rdb_total)

        # 8. Encoder Stem
        d_s0_pre = np.where(self.s0_pre > 0, d_s0, d_s0 * 0.2)
        dx = self.stem.backward(d_s0_pre)
        return dx

    def get_trainable_layers(self):
        """Collect all trainable Conv2D layers."""
        layers = [
            self.stem, self.down1, self.down2,
            self.rdb0.conv1, self.rdb0.conv2, self.rdb0.conv3,
            self.rdb1.conv1, self.rdb1.conv2, self.rdb1.conv3,
            self.conv_up1, self.fuse1, self.conv_up2, self.fuse2,
            self.mask_head, self.conf_conv, self.conf_fc
        ]
        return layers

    def get_weights(self):
        """Export all trainable weights as a state dictionary."""
        state = {}
        for attr, val in self.__dict__.items():
            if isinstance(val, Conv2D):
                state[f"{attr}.W"] = val.W
                if val.use_bias:
                    state[f"{attr}.b"] = val.b
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
            elif isinstance(val, CBAMBlock):
                state[f"{attr}.ca.fc1.W"] = val.ca.fc1.W
                state[f"{attr}.ca.fc2.W"] = val.ca.fc2.W
                state[f"{attr}.sa.conv.W"] = val.sa.conv.W
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
                elif len(parts) == 4:
                    sub1 = getattr(mod, parts[1], None)
                    if sub1:
                        sub2 = getattr(sub1, parts[2], None)
                        if sub2:
                            setattr(sub2, parts[3], v.copy())
