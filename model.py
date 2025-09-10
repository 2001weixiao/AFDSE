import torch.nn as nn
import torch
import pywt
import torch.nn.functional as F
from torch.nn import LayerNorm

class AFDSE(nn.Module):
    def __init__(self, num_classes, encoder_dim, d1, d2,wavelet):
        super(AFDSE, self).__init__()

        self.conv_hsi = nn.Conv2d(d1, 64, kernel_size=3, stride=1, padding=1)
        self.conv_lidar = nn.Conv2d(d2, 64, kernel_size=3, stride=1, padding=1)

        self.wavelet = wavelet
        self.level = 1

        self.enhance = nn.Sequential(
            AFE(64),
            SE(
                dim=64,
                num_heads=4,
                BasicConv=BasicConv,
                sparse_threshold=0.3,
                high_pass=False
            )
        )

        self.Gate = Gate(64, 64, BasicConv=BasicConv)

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(encoder_dim * 2, int(encoder_dim / 2))
        self.trans_norm = nn.LayerNorm(encoder_dim * 2)

        self.Classifier = nn.Sequential(
            nn.Linear(int(encoder_dim / 2), num_classes),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(num_classes),
        )

        self.cbam_module = CBAM(num_channels=encoder_dim * 2, reduction_ratio=16, kernel_size=7)

        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, hsi_inputs, lidar_inputs):
        hsi = self.conv_hsi(hsi_inputs)
        lidar = self.conv_lidar(lidar_inputs)
        hsi_LL, hsi_H = self._dwt(hsi_inputs)
        lidar_LL, lidar_H = self._dwt(lidar_inputs)

        hsi_low = self.enhance(self.conv_hsi(hsi_LL))
        lidar_low = self.enhance(self.conv_lidar(lidar_LL))
        hsi_high = [self.enhance(self.conv_hsi(band)) for band in hsi_H]
        lidar_high = [self.enhance(self.conv_lidar(band)) for band in lidar_H]

        hsi_recon = self._idwt(hsi_low, hsi_high, hsi_inputs.shape)
        lidar_recon = self._idwt(lidar_low, lidar_high, lidar_inputs.shape)

        fused_hsi = self.Gate(hsi, hsi_recon)
        fused_lidar = self.Gate(lidar, lidar_recon)

        batch_size = fused_hsi.size(0)
        hsi_flat = fused_hsi.view(batch_size, -1)  # [B, D1]
        lidar_flat = fused_lidar.view(batch_size, -1)  # [B, D2]
        hsi_centered = hsi_flat - hsi_flat.mean(dim=0, keepdim=True)
        lidar_centered = lidar_flat - lidar_flat.mean(dim=0, keepdim=True)
        covariance = torch.mm(hsi_centered.T, lidar_centered) / (batch_size - 1)  # [D1, D2]
        con_loss = torch.sum(covariance ** 2) / (hsi_flat.size(1) * lidar_flat.size(1))


        feature = torch.cat([fused_hsi, fused_lidar], dim=1)
        feature = self.cbam_module(feature)

        x_t = self.pooling(feature).flatten(1)
        final_class = self.conv_cls_head(x_t)

        final_class = self.Classifier(final_class)
        final_class = final_class.view(hsi_inputs.shape[0], -1)

        return final_class, con_loss
    def _dwt(self, x):
        x_np = x.cpu().numpy()
        coeffs = pywt.wavedec2(x_np, wavelet=self.wavelet, level=self.level, mode='symmetric')
        LL, (LH, HL, HH) = coeffs[0], coeffs[1:][0]

        LL = torch.from_numpy(LL).to(x.device)
        LH = torch.from_numpy(LH).to(x.device)
        HL = torch.from_numpy(HL).to(x.device)
        HH = torch.from_numpy(HH).to(x.device)

        return LL, [LH, HL, HH]

    def _idwt(self, LL, H, original_shape):
        LL_np = LL.detach().cpu().numpy()
        H_np = [h.detach().cpu().numpy() for h in H]
        coeffs = [LL_np, H_np]
        recon = pywt.waverec2(coeffs, wavelet=self.wavelet, mode='symmetric')
        recon = torch.from_numpy(recon).to(LL.device)

        recon = recon[:, :, :original_shape[-2], :original_shape[-1]]

        return recon

#-----------------BasicConv-----------------------------
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1, high_pass=False, sparse_activation=False):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        self.high_pass = high_pass
        self.sparse_activation = sparse_activation

        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups)
            )
        else:
            if high_pass:
                layers.append(
                    nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=bias, groups=groups)
                )
                layers.append(
                    nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=bias, groups=out_channel)
                )
            else:
                layers.append(
                    nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups)
                )

        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if sparse_activation:
                layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

#-----------------AFE Module-----------------------------
class AFE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.high_pass_enhancement = self.AdaptiveHighPassEnhancement(in_channels)

    class AdaptiveHighPassEnhancement(nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            # S
            self.info_branch = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1)
            )

            # N
            self.noise_branch = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.AdaptiveAvgPool2d(1)
            )

            # R
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 2, 1),
                nn.GELU(),
                nn.Conv2d(in_channels // 2, in_channels, 1),
                nn.Sigmoid()
            )

            # W
            self.fusion = nn.Conv2d(in_channels * 3, in_channels, 1)

            # Laplacian convolution kernels
            self.laplacian_kernel = nn.Parameter(
                torch.tensor([[0, -1, 0],
                              [-1, 4, -1],
                              [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1),
                requires_grad=True
            )

        def forward(self, x):
            info = self.info_branch(x)
            noise = self.noise_branch(x)
            attention = self.attention(x)

            combined = torch.cat([info, noise, attention], dim=1)
            weights = torch.sigmoid(self.fusion(combined))

            laplacian = F.conv2d(x, self.laplacian_kernel, padding=1, groups=x.size(1))
            enhanced = x + laplacian * weights

            return enhanced

    def forward(self, x):
        return self.high_pass_enhancement(x)
    
#-----------------SE Module-----------------------------
class SE(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=True,
                 LayerNorm_type='WithBias', BasicConv=BasicConv,
                 sparse_threshold=0.3, high_pass=True):
        super().__init__()
        self.dim = dim
        self.sparse_threshold = sparse_threshold

        # norm
        self.norm1 = AdaptiveSparseNorm(dim, LayerNorm_type)
        self.norm2 = AdaptiveSparseNorm(dim, LayerNorm_type)

        # sparse attention mechanism
        self.attn = SparseAttention(
            dim=dim,
            num_heads=num_heads,
            bias=bias,
            high_pass=high_pass
        )

        # low pass conv
        self.low_pass_conv = BasicConv(
            in_channel=dim,
            out_channel=dim,
            kernel_size=3,
            stride=1,
            relu=False,
            high_pass=False
        )

        # feature enhancement network
        self.low_freq_enhancement = Feature_Enhancement_Network(dim, sparse_threshold)

        # Feedforward Neural Network
        self.ffn = Feedforward_Neural_Network(
            dim=dim,
            ffn_expansion_factor=ffn_expansion_factor,
            BasicConv=BasicConv,
            sparse_threshold=sparse_threshold
        )

    def forward(self, x):
        x = x + self._sparse_gate(self.attn(self.norm1(x)))

        low_freq_feature = self.low_pass_conv(x)
        low_freq_compensated = self.low_freq_enhancement(low_freq_feature)

        x = x + low_freq_compensated
        x = x + self._sparse_gate(self.ffn(self.norm2(x)))

        return x

    def _sparse_gate(self, x):
        mask = torch.sigmoid((x.abs() - self.sparse_threshold) * 10)
        return x * mask

class Feature_Enhancement_Network(nn.Module):
    def __init__(self, dim, sparse_threshold):
        super().__init__()
        self.dim = dim
        self.sparse_threshold = sparse_threshold

        self.compensation_network = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        )

        self.sparse_gate = SparseGate(sparse_threshold)

    def forward(self, x):
        compensated = self.compensation_network(x)
        return self.sparse_gate(compensated)


class SparseGate(nn.Module):
    def __init__(self, sparse_threshold):
        super().__init__()
        self.sparse_threshold = sparse_threshold

    def forward(self, x):
        mask = torch.sigmoid((x.abs() - self.sparse_threshold) * 10)
        return x * mask
class AdaptiveSparseNorm(nn.Module):
    def __init__(self, dim, norm_type):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-5)
        self.sparse_proj = nn.Linear(dim, dim)

    def forward(self, x):
        original_shape = x.shape  # [batch_size, channels, height, width]
        batch_size, channels, height, width = original_shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, channels)
        x_norm = self.norm(x)

        x_norm_reshaped = x_norm.view(batch_size, height, width, channels)
        avg_pooled = x_norm_reshaped.mean(dim=[1, 2])
        sparse_weight = torch.sigmoid(self.sparse_proj(avg_pooled))

        sparse_weight = sparse_weight.unsqueeze(1).unsqueeze(1).expand(-1, height, width, -1)
        x_norm = x_norm_reshaped * sparse_weight
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()

        return x_norm
class SparseAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, high_pass):
        super().__init__()
        self.num_heads = num_heads
        self.high_pass = high_pass

        self.local_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            bias=bias,
            batch_first=True
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x_flat = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        local_mask = self._create_local_mask(H, W, window_size=3, x=x)
        attn_out, _ = self.local_attn(
            query=x_flat,
            key=x_flat,
            value=x_flat,
            attn_mask=local_mask
        )
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)

        return attn_out

    def _create_local_mask(self, H, W, window_size=3, x=None):
        mask = torch.zeros((H * W, H * W), dtype=torch.bool)
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < H and 0 <= nj < W:
                            n_idx = ni * W + nj
                            mask[idx, n_idx] = True
        if x is not None:
            return mask.to(x.device)
        else:
            return mask


class Feedforward_Neural_Network(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, BasicConv, sparse_threshold):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)

        self.conv1 = BasicConv(
            in_channel=dim,
            out_channel=hidden_dim,
            kernel_size=3,
            stride=1,
            relu=False,
            high_pass=False
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Sigmoid()
        )

        self.conv2 = BasicConv(
            in_channel=hidden_dim,
            out_channel=dim,
            kernel_size=3,
            stride=1,
            relu=False,
            sparse_activation=True
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        gate = self.gate(x)
        x = self.conv1(x) * gate
        x = self.conv2(x)

        return self.alpha * x

#-----------------Gate-----------------------------
class Gate(nn.Module):
    def __init__(self, in_channel, out_channel, BasicConv=BasicConv):
        super(Gate, self).__init__()
        self.conv_max = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.conv_mid = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

        self.conv1 =BasicConv(in_channel, out_channel = 64, kernel_size=1, stride=1, relu=True)
        self.conv2 = BasicConv(in_channel=64, out_channel=64, kernel_size=1, stride=1, relu=True)

    def forward(self, x_max,x_mid):

        y_max = x_max +x_mid
        y_max = self.conv1(y_max)

        x_max = self.conv_max(x_max)
        x_mid = self.conv_mid(x_mid)

        x =F.tanh(x_mid) * x_max
        x = self.conv2(x)

        return x+y_max

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(num_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x