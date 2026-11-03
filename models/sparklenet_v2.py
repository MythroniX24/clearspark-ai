"""
ClearSpark AI SparkleNet v2.0 — PyTorch Reference Architecture
Engineered for Kaggle Dual T4 GPU Training on 20,000+ images.

Features:
1. MultiScaleAttentionUNet (WAL-Net v2): Scale-invariant watermark localization (24px - 160px),
   Continuous scale regression, ASPP, and CBAM Dual Attention.
2. ContextualGenerativeInpainter (CIS-Net v2): Gated Convolutions, Multi-rate Dilations (r=1,2,4,8),
   Residual Dense Refinement, and Contextual Texture Synthesis.
3. Automatic weight export to sparklenet_web_weights.json for client-side web inference.
"""

import math
import json
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    # ── 1. Attention & Convolutional Primitives ──────────────────────────────

    class ChannelAttention(nn.Module):
        """Channel Attention Module (CAM) for focusing on watermark spectral signatures."""
        def __init__(self, in_channels, reduction=8):
            super().__init__()
            reduced = max(4, in_channels // reduction)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.mlp = nn.Sequential(
                nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduced, in_channels, kernel_size=1, bias=False)
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = self.mlp(self.avg_pool(x))
            max_out = self.mlp(self.max_pool(x))
            return x * self.sigmoid(avg_out + max_out)

    class SpatialAttention(nn.Module):
        """Spatial Attention Module (SAM) for sub-pixel localization of the 4-pointed sparkle."""
        def __init__(self, kernel_size=7):
            super().__init__()
            padding = kernel_size // 2
            self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
            return x * scale

    class CBAMBlock(nn.Module):
        """Convolutional Block Attention Module combining Channel and Spatial Attention."""
        def __init__(self, in_channels, reduction=8):
            super().__init__()
            self.ca = ChannelAttention(in_channels, reduction)
            self.sa = SpatialAttention(kernel_size=7)

        def forward(self, x):
            return self.sa(self.ca(x))

    class ResidualDenseBlock(nn.Module):
        """Residual Dense Block (RDB) for sub-pixel gradient reuse."""
        def __init__(self, in_channels, growth_rate=12):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, in_channels, kernel_size=3, padding=1)
            self.act = nn.LeakyReLU(0.1, inplace=True)

        def forward(self, x):
            c1 = self.act(self.conv1(x))
            c2 = self.act(self.conv2(torch.cat([x, c1], dim=1)))
            c3 = self.conv3(torch.cat([x, c1, c2], dim=1))
            return x + c3

    class GatedConv2d(nn.Module):
        """
        Gated Convolution for Inpainting.
        Learns dynamic gating to ignore occluded watermark pixels and propagate clean background context.
        """
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
            super().__init__()
            self.conv_f = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
            self.conv_g = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)

        def forward(self, x):
            f = F.leaky_relu(self.conv_f(x), 0.1)
            g = torch.sigmoid(self.conv_g(x))
            return f * g


    # ── 2. WAL-Net v2 (Watermark Attention Localization Network) ─────────────

    class MultiScaleAttentionUNet(nn.Module):
        """
        Scale-Invariant Watermark Localization Network.
        Processes patches from 24px to 160px with Residual Dense Blocks,
        CBAM bottleneck attention, and multi-task regression heads.
        """
        def __init__(self, in_channels=3, base_channels=16):
            super().__init__()
            self.base = base_channels

            # Stem
            self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
            self.stem_act = nn.LeakyReLU(0.1, inplace=True)
            self.rdb0 = ResidualDenseBlock(base_channels, growth_rate=12)

            # Downsample 1
            self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
            self.down1_act = nn.LeakyReLU(0.1, inplace=True)
            self.rdb1 = ResidualDenseBlock(base_channels * 2, growth_rate=16)

            # Downsample 2
            self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
            self.down2_act = nn.LeakyReLU(0.1, inplace=True)

            # Bottleneck with CBAM Attention
            self.bottleneck = CBAMBlock(base_channels * 4)

            # Multi-Task Regression Heads (Confidence + Continuous BBox)
            self.conf_conv = nn.Conv2d(base_channels * 4, 16, kernel_size=1)
            self.conf_act = nn.LeakyReLU(0.1, inplace=True)
            self.conf_gap = nn.AdaptiveAvgPool2d(1)
            self.conf_fc = nn.Conv2d(16, 5, kernel_size=1) # [conf, cx, cy, w, h]

            # Upsample 1 & Skip Fusion
            self.conv_up1 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
            self.up1_act = nn.LeakyReLU(0.1, inplace=True)
            self.fuse1 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
            self.fuse1_act = nn.LeakyReLU(0.1, inplace=True)

            # Upsample 2 & Skip Fusion
            self.conv_up2 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
            self.up2_act = nn.LeakyReLU(0.1, inplace=True)
            self.fuse2 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
            self.fuse2_act = nn.LeakyReLU(0.1, inplace=True)

            # Pixel Probability Mask Head
            self.mask_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

        def forward(self, x):
            # Encoder
            s0 = self.rdb0(self.stem_act(self.stem(x)))
            d1 = self.rdb1(self.down1_act(self.down1(s0)))
            d2 = self.down2_act(self.down2(d1))
            b = self.bottleneck(d2)

            # Confidence & Continuous Scale BBox
            c_feat = self.conf_act(self.conf_conv(b))
            c_gap = self.conf_gap(c_feat)
            c_out = torch.sigmoid(self.conf_fc(c_gap)) # (N, 5, 1, 1)

            conf = c_out[:, 0:1, 0, 0] # (N, 1)
            bbox = c_out[:, 1:5, 0, 0] # (N, 4) -> [cx, cy, w, h]

            # Decoder (robust spatial size matching for arbitrary input dimensions)
            u1 = self.up1_act(self.conv_up1(F.interpolate(b, size=d1.shape[2:], mode='nearest')))
            f1 = self.fuse1_act(self.fuse1(torch.cat([u1, d1], dim=1)))

            u2 = self.up2_act(self.conv_up2(F.interpolate(f1, size=s0.shape[2:], mode='nearest')))
            f2 = self.fuse2_act(self.fuse2(torch.cat([u2, s0], dim=1)))

            mask = torch.sigmoid(self.mask_head(f2))

            return {
                'mask': mask,
                'confidence': conf,
                'bbox': bbox
            }


    # ── 3. CIS-Net v2 (Contextual Inpainting & Background Synthesis) ──────────

    class ContextualGenerativeInpainter(nn.Module):
        """
        Deep Gated Contextual Inpainting Network.
        Takes 4-channel input: (3-channel masked RGB + 1-channel binary mask).
        Synthesizes photorealistic background textures in place of eradicated watermark.
        """
        def __init__(self, in_channels=4, base_channels=16):
            super().__init__()
            self.base = base_channels

            # Gated Convolutions with wide receptive field
            self.gate1 = GatedConv2d(in_channels, base_channels, kernel_size=5, stride=1, padding=2)
            self.gate2 = GatedConv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1)

            # Dilated Gated Convolutions (capturing wider context surrounding the watermark)
            self.dilated1 = GatedConv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=2, dilation=2)
            self.dilated2 = GatedConv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=4, dilation=4)

            # Residual Texture Synthesis Block
            self.rdb = ResidualDenseBlock(base_channels * 2, growth_rate=16)

            # Output Refinement
            self.gate3 = GatedConv2d(base_channels * 2, base_channels, kernel_size=3, stride=1, padding=1)
            self.out_conv = nn.Conv2d(base_channels, 3, kernel_size=3, stride=1, padding=1)

        def forward(self, img, mask):
            """
            Args:
                img: (N, 3, H, W) normalized image in [0, 1]
                mask: (N, 1, H, W) binary watermark mask
            Returns:
                inpainted: (N, 3, H, W) complete photorealistic reconstructed image
            """
            masked_img = img * (1.0 - mask)
            x = torch.cat([masked_img, mask], dim=1) # (N, 4, H, W)

            h1 = self.gate1(x)
            h2 = self.gate2(h1)

            # Dilated context integration
            d1 = self.dilated1(h2)
            d2 = self.dilated2(d1)

            # Residual feature refinement
            r = self.rdb(d2)

            # Fusion with skip connection
            h3 = self.gate3(r + h2)
            synth_rgb = torch.sigmoid(self.out_conv(h3))

            # Alpha-blended composite output
            inpainted = img * (1.0 - mask) + synth_rgb * mask
            return inpainted, synth_rgb


    # ── 4. Web Weights Exporter ──────────────────────────────────────────────

    def export_pytorch_to_web_json(wal_net, cis_net, out_path, tpl48=None, tpl96=None):
        """
        Exports PyTorch trained models into 100% compliant sparklenet_web_weights.json.
        Compatible with browser client-side watermark-cleaner.js.
        """
        bundle = {'wal_net': {}, 'cis_net': {}}

        def export_layer(module, prefix, dest_dict):
            if isinstance(module, nn.Conv2d):
                W = module.weight.detach().cpu().numpy()
                b = module.bias.detach().cpu().numpy() if module.bias is not None else np.zeros(module.out_channels, dtype=np.float32)
                dest_dict[prefix] = {
                    'shape': list(W.shape),
                    'data': [float(np.round(x, 5)) for x in W.flatten()],
                    'bias': [float(np.round(x, 5)) for x in b.flatten()]
                }
            elif isinstance(module, GatedConv2d):
                export_layer(module.conv_f, f"{prefix}.conv_f", dest_dict)
                export_layer(module.conv_g, f"{prefix}.conv_g", dest_dict)
            elif isinstance(module, ResidualDenseBlock):
                export_layer(module.conv1, f"{prefix}.conv1", dest_dict)
                export_layer(module.conv2, f"{prefix}.conv2", dest_dict)
                export_layer(module.conv3, f"{prefix}.conv3", dest_dict)
            elif isinstance(module, CBAMBlock):
                # Channel attention MLP
                export_layer(module.ca.mlp[0], f"{prefix}.ca.fc1", dest_dict)
                export_layer(module.ca.mlp[2], f"{prefix}.ca.fc2", dest_dict)
                # Spatial attention Conv
                export_layer(module.sa.conv, f"{prefix}.sa.conv", dest_dict)

        # Export WAL-Net layers
        export_layer(wal_net.stem, 'stem', bundle['wal_net'])
        export_layer(wal_net.rdb0, 'rdb0', bundle['wal_net'])
        export_layer(wal_net.down1, 'down1', bundle['wal_net'])
        export_layer(wal_net.rdb1, 'rdb1', bundle['wal_net'])
        export_layer(wal_net.down2, 'down2', bundle['wal_net'])
        export_layer(wal_net.bottleneck, 'bottleneck', bundle['wal_net'])
        export_layer(wal_net.conf_conv, 'conf_conv', bundle['wal_net'])
        export_layer(wal_net.conf_fc, 'conf_fc', bundle['wal_net'])
        export_layer(wal_net.conv_up1, 'conv_up1', bundle['wal_net'])
        export_layer(wal_net.fuse1, 'fuse1', bundle['wal_net'])
        export_layer(wal_net.conv_up2, 'conv_up2', bundle['wal_net'])
        export_layer(wal_net.fuse2, 'fuse2', bundle['wal_net'])
        export_layer(wal_net.mask_head, 'mask_head', bundle['wal_net'])

        # Export CIS-Net layers
        export_layer(cis_net.gate1, 'gate1', bundle['cis_net'])
        export_layer(cis_net.gate2, 'gate2', bundle['cis_net'])
        export_layer(cis_net.dilated1, 'dilated1', bundle['cis_net'])
        export_layer(cis_net.dilated2, 'dilated2', bundle['cis_net'])
        export_layer(cis_net.rdb, 'rdb', bundle['cis_net'])
        export_layer(cis_net.gate3, 'gate3', bundle['cis_net'])
        export_layer(cis_net.out_conv, 'out_conv', bundle['cis_net'])

        if tpl48 is not None:
            bundle['template_48'] = tpl48.tolist() if isinstance(tpl48, np.ndarray) else tpl48
        if tpl96 is not None:
            bundle['template_96'] = tpl96.tolist() if isinstance(tpl96, np.ndarray) else tpl96

        dir_name = os.path.dirname(out_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(bundle, f)

        size_kb = os.path.getsize(out_path) / 1024.0
        print(f"✅ Successfully exported web weights to {out_path} ({size_kb:.1f} KB, {len(bundle['wal_net'])} WAL layers, {len(bundle['cis_net'])} CIS layers)")
        return bundle

else:
    # ── Non-PyTorch Fallback Stubs ──────────────────────────────────────────
    class MultiScaleAttentionUNet:
        """PyTorch fallback stub when torch is not installed."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PyTorch is not installed in this environment. "
                "MultiScaleAttentionUNet requires PyTorch. "
                "Please install torch via 'pip install torch' or run inside a GPU runtime (e.g., Kaggle, Colab)."
            )

    class ContextualGenerativeInpainter:
        """PyTorch fallback stub when torch is not installed."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PyTorch is not installed in this environment. "
                "ContextualGenerativeInpainter requires PyTorch. "
                "Please install torch via 'pip install torch' or run inside a GPU runtime (e.g., Kaggle, Colab)."
            )

    class ChannelAttention:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for ChannelAttention.")

    class SpatialAttention:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for SpatialAttention.")

    class CBAMBlock:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for CBAMBlock.")

    class ResidualDenseBlock:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for ResidualDenseBlock.")

    class GatedConv2d:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for GatedConv2d.")

    def export_pytorch_to_web_json(*args, **kwargs):
        raise RuntimeError("PyTorch is required to export weights from PyTorch models.")

