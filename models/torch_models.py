"""
PyTorch Reference Implementations for WAL-Net & CIS-Net
Use this module for GPU-accelerated cloud training (CUDA / MPS / ROCm)
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class GatedConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            super().__init__()
            self.conv_f = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv_g = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        def forward(self, x):
            f = F.elu(self.conv_f(x))
            g = torch.sigmoid(self.conv_g(x))
            return f * g

    class PyTorchSparkleAttentionUNet(nn.Module):
        def __init__(self, in_channels=3, base_channels=32):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.down1 = nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.down2 = nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.up1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(base_channels * 4, base_channels, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.mask_head = nn.Sequential(
                nn.Conv2d(base_channels * 2, 1, 3, padding=1),
                nn.Sigmoid()
            )

        def forward(self, x):
            s0 = self.stem(x)
            d1 = self.down1(s0)
            d2 = self.down2(d1)
            u1 = self.up1(d2)
            c1 = torch.cat([u1, d1], dim=1)
            u2 = self.up2(c1)
            c2 = torch.cat([u2, s0], dim=1)
            mask = self.mask_head(c2)
            return mask
else:
    class PyTorchSparkleAttentionUNet:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not installed in the current environment.")
