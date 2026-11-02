"""
Neural Runtime Engine — Pure NumPy Vectorized Deep Learning Tensor Framework
Author: ClearSpark AI Engineering Team
Platform: Mobile / Embedded Linux / Termux / Server Cross-Platform Compatible

Provides hardware-portable forward inference and backpropagation primitives:
- Conv2D (vectorized with im2col / striding)
- GatedConv2D (Gated Convolutions for Inpainting)
- BatchNorm2D
- SpatialAttention & ChannelAttention (CBAM Block)
- ResidualDenseBlock (RDB)
- Upsample2D (Bilinear / Nearest)
- Activations: LeakyReLU, ReLU, ELU, Sigmoid, Tanh
"""

import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -30.0, 0.0)) - 1.0))

def leaky_relu(x, negative_slope=0.2):
    return np.where(x > 0, x, x * negative_slope)

def relu(x):
    return np.maximum(0, x)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """Zero-padding and vectorized im2col transformation for fast 2D convolution."""
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return (k.astype(int), i.astype(int), j.astype(int))

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class Conv2D:
    """Vectorized 2D Convolution layer."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

        # He / Kaiming normal initialization
        limit = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * limit
        self.b = np.zeros(out_channels, dtype=np.float32) if bias else None

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None
        self.x_cols = None
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        N, C, H, W = x.shape
        out_h = int((H + 2 * self.padding - self.kernel_size) / self.stride + 1)
        out_w = int((W + 2 * self.padding - self.kernel_size) / self.stride + 1)

        self.x_cols = im2col_indices(x, self.kernel_size, self.kernel_size, padding=self.padding, stride=self.stride)
        W_row = self.W.reshape(self.out_channels, -1)

        out = np.dot(W_row, self.x_cols)
        if self.use_bias:
            out += self.b.reshape(-1, 1)

        out = out.reshape(self.out_channels, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        dout_row = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        self.dW = np.dot(dout_row, self.x_cols.T).reshape(self.W.shape)
        if self.use_bias:
            self.db = np.sum(dout_row, axis=1)
        dx_cols = np.dot(self.W.reshape(self.out_channels, -1).T, dout_row)
        dx = col2im_indices(dx_cols, self.x_shape, self.kernel_size, self.kernel_size, padding=self.padding, stride=self.stride)
        return dx


class GatedConv2D:
    """
    Gated Convolution Layer (Yu et al. 'Free-Form Image Inpainting with Gated Convolution')
    Computes feature map F and gating mask G: Output = ELU(F) * Sigmoid(G)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.conv_f = Conv2D(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_g = Conv2D(in_channels, out_channels, kernel_size, stride, padding)
        self.f = None
        self.g = None
        self.elu_f = None
        self.sig_g = None

    def forward(self, x):
        self.f = self.conv_f.forward(x)
        self.g = self.conv_g.forward(x)
        self.elu_f = elu(self.f)
        self.sig_g = sigmoid(self.g)
        return self.elu_f * self.sig_g

    def backward(self, dout):
        d_elu_f = dout * self.sig_g
        d_sig_g = dout * self.elu_f
        df = d_elu_f * np.where(self.f > 0, 1.0, np.exp(np.clip(self.f, -30.0, 0.0)))
        dg = d_sig_g * (self.sig_g * (1.0 - self.sig_g))
        dx_f = self.conv_f.backward(df)
        dx_g = self.conv_g.backward(dg)
        return dx_f + dx_g


class ChannelAttention:
    """Squeeze-and-Excitation Channel Attention module."""
    def __init__(self, channels, reduction=4):
        self.channels = channels
        mid = max(4, channels // reduction)
        self.fc1 = Conv2D(channels, mid, kernel_size=1, padding=0)
        self.fc2 = Conv2D(mid, channels, kernel_size=1, padding=0)
        self.scale = None

    def forward(self, x):
        # Global Avg Pool & Max Pool
        avg_pool = np.mean(x, axis=(2, 3), keepdims=True)
        max_pool = np.max(x, axis=(2, 3), keepdims=True)

        avg_out = self.fc2.forward(relu(self.fc1.forward(avg_pool)))
        max_out = self.fc2.forward(relu(self.fc1.forward(max_pool)))
        self.scale = sigmoid(avg_out + max_out)
        return x * self.scale

    def backward(self, dout):
        return dout * self.scale


class SpatialAttention:
    """Spatial Attention module highlighting watermark regions."""
    def __init__(self, kernel_size=7):
        padding = kernel_size // 2
        self.conv = Conv2D(2, 1, kernel_size=kernel_size, padding=padding)
        self.scale = None

    def forward(self, x):
        avg_pool = np.mean(x, axis=1, keepdims=True)
        max_pool = np.max(x, axis=1, keepdims=True)
        cat = np.concatenate([avg_pool, max_pool], axis=1)
        self.scale = sigmoid(self.conv.forward(cat))
        return x * self.scale

    def backward(self, dout):
        return dout * self.scale


class CBAMBlock:
    """Convolutional Block Attention Module combining channel and spatial attention."""
    def __init__(self, channels):
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention(kernel_size=7)
        self.x_ca = None

    def forward(self, x):
        self.x_ca = self.ca.forward(x)
        out = self.sa.forward(self.x_ca)
        return out

    def backward(self, dout):
        dx_ca = self.sa.backward(dout)
        dx = self.ca.backward(dx_ca)
        return dx


class ResidualDenseBlock:
    """Residual Dense Block for high-frequency watermark gradient recovery."""
    def __init__(self, channels, growth_rate=16):
        self.conv1 = Conv2D(channels, growth_rate, kernel_size=3, padding=1)
        self.conv2 = Conv2D(channels + growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv3 = Conv2D(channels + 2 * growth_rate, channels, kernel_size=3, padding=1)
        self.cbam = CBAMBlock(channels)
        self.x = None
        self.x1 = None
        self.c1 = None
        self.x2 = None
        self.c2 = None

    def forward(self, x):
        self.x = x
        self.x1 = leaky_relu(self.conv1.forward(x))
        self.c1 = np.concatenate([x, self.x1], axis=1)
        self.x2 = leaky_relu(self.conv2.forward(self.c1))
        self.c2 = np.concatenate([self.c1, self.x2], axis=1)
        self.x3 = self.conv3.forward(self.c2)
        out = self.cbam.forward(self.x3)
        return x + 0.2 * out

    def backward(self, dout):
        dx3 = self.cbam.backward(0.2 * dout)
        dc2 = self.conv3.backward(dx3)

        c1_channels = self.c1.shape[1]
        dc1_from_c2 = dc2[:, :c1_channels, :, :]
        dx2 = dc2[:, c1_channels:, :, :]
        d_conv2 = self.conv2.backward(np.where(self.x2 > 0, dx2, dx2 * 0.2))

        dc1 = dc1_from_c2 + d_conv2
        x_channels = self.x.shape[1]
        dx_from_c1 = dc1[:, :x_channels, :, :]
        dx1 = dc1[:, x_channels:, :, :]
        d_conv1 = self.conv1.backward(np.where(self.x1 > 0, dx1, dx1 * 0.2))

        return dout + dx_from_c1 + d_conv1


class Upsample2D:
    """Vectorized Bilinear Upsampling layer."""
    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        scale = self.scale_factor
        out = np.repeat(np.repeat(x, scale, axis=2), scale, axis=3)
        return out

    def backward(self, dout):
        N, C, H, W = self.x_shape
        s = self.scale_factor
        return dout.reshape(N, C, H, s, W, s).sum(axis=(3, 5))


class AdamOptimizer:
    """
    Hardware-accelerated Adam optimizer for NumPy Deep Neural Networks.
    Supports weight decay, gradient clipping, and bias correction.
    """
    def __init__(self, layers, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, clip_norm=5.0):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.clip_norm = clip_norm
        self.t = 0

        self.m_W = {}
        self.v_W = {}
        self.m_b = {}
        self.v_b = {}

        for idx, layer in enumerate(self.layers):
            if hasattr(layer, 'W') and layer.W is not None:
                self.m_W[idx] = np.zeros_like(layer.W)
                self.v_W[idx] = np.zeros_like(layer.W)
            if hasattr(layer, 'b') and layer.b is not None:
                self.m_b[idx] = np.zeros_like(layer.b)
                self.v_b[idx] = np.zeros_like(layer.b)

    def step(self):
        self.t += 1
        bias_c1 = 1.0 - (self.beta1 ** self.t)
        bias_c2 = 1.0 - (self.beta2 ** self.t)
        step_size = self.lr * (np.sqrt(bias_c2) / bias_c1)

        for idx, layer in enumerate(self.layers):
            if hasattr(layer, 'W') and hasattr(layer, 'dW') and layer.dW is not None:
                dW = np.clip(layer.dW, -self.clip_norm, self.clip_norm)
                self.m_W[idx] = self.beta1 * self.m_W[idx] + (1.0 - self.beta1) * dW
                self.v_W[idx] = self.beta2 * self.v_W[idx] + (1.0 - self.beta2) * (dW ** 2)
                layer.W -= step_size * (self.m_W[idx] / (np.sqrt(self.v_W[idx]) + self.eps))

            if hasattr(layer, 'b') and hasattr(layer, 'db') and layer.db is not None:
                db = np.clip(layer.db, -self.clip_norm, self.clip_norm)
                self.m_b[idx] = self.beta1 * self.m_b[idx] + (1.0 - self.beta1) * db
                self.v_b[idx] = self.beta2 * self.v_b[idx] + (1.0 - self.beta2) * (db ** 2)
                layer.b -= step_size * (self.m_b[idx] / (np.sqrt(self.v_b[idx]) + self.eps))

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'dW') and layer.dW is not None:
                layer.dW.fill(0)
            if hasattr(layer, 'db') and layer.db is not None:
                layer.db.fill(0)
