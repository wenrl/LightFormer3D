import torch
import torch.nn as nn
import torch.nn.functional as F

def channel_shuffle(x, groups):
    batch_size, num_channels, depth, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, depth, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, depth, height, width)

    return x


class MSA(nn.Module):
    def __init__(self, channels, reduction=8, use_residual=True, CA = True, SA = True):
        super(MSA, self).__init__()
        self.channels = channels
        self.reduction = reduction
        # self.kernel_sizes = kernel_sizes
        self.use_residual = use_residual
        self.dilations = [1, 2, 1]
        self.ca = CA
        self.sa = SA

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, padding=1, dilation=1, groups=1, bias=False),
            nn.Conv3d(4, 1, kernel_size=1, bias=False),
            nn.InstanceNorm3d(4),
            nn.GELU())
        self.layer2 = nn.Sequential(
            nn.Conv3d(2, 8, kernel_size=3, padding=2, dilation=2, groups=2, bias=False),
            nn.Conv3d(8, 2, kernel_size=1, bias=False),
            nn.InstanceNorm3d(8),
            nn.GELU())
        self.layer3 = nn.Sequential(
            nn.Conv3d(3, 12, kernel_size=5, padding=2, dilation=1, groups=3, bias=False),
            nn.Conv3d(12, 3, kernel_size=1, bias=False),
            nn.InstanceNorm3d(12),
            nn.GELU())

        self.spatial_fusion = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=3, groups=4, padding=1, bias=False),
            nn.Conv3d(4, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )


    def forward1(self, x):

        channel_weights = self.channel_attention(x)
        channel_refined = x * channel_weights + x

        spatial_input = torch.mean(channel_refined, dim=1, keepdim=True)

        out1 = torch.cat([self.layer1(spatial_input), spatial_input], dim=1)
        out2 = torch.cat([self.layer2(out1), spatial_input], dim=1)
        out3 = torch.cat([self.layer3(out2), spatial_input], dim=1)
        spatial_attention = self.spatial_fusion(out3)

        output = channel_refined * spatial_attention + channel_refined

        return output
    
    def forward2(self, x):
        channel_weights = self.channel_attention(x)
        output = x * channel_weights + x

        return output
    
    def forward3(self, x):
        spatial_input = torch.mean(x, dim=1, keepdim=True)
        out1 = torch.cat([self.branch1(spatial_input), spatial_input], dim=1)
        out2 = torch.cat([self.branch2(out1), spatial_input], dim=1)
        out3 = torch.cat([self.branch3(out2), spatial_input], dim=1)

        spatial_attention = self.spatial_fusion(out3)

        output = x * spatial_attention + x

        return output
    
    def forward(self, x):
        if self.ca == False:
            output = self.forward3(x)
        elif self.sa == False:
            output = self.forward2(x)
        else:
            output = self.forward1(x)
        return output

class LocalGlobalFusion3D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=3):
        super(LocalGlobalFusion3D, self).__init__()
        self.channels = channels

        self.gate = Gated3D(channels, kernel_size)
        self.fusion = Fuse3D(channels*2, kernel_size)

        
    def forward(self, x1, x2):

        fused = self.gate_fusion(x1, x2)
        enhanced = self.enhancement(fused)
        return enhanced

class Gated3D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(Gated3D, self).__init__()
        padding = kernel_size // 2
        self.channels = channels

        self.norm = nn.InstanceNorm3d(channels)
        self.gate = nn.Sequential(
            nn.Conv3d(channels * 2, channels * 2, kernel_size, padding=padding, groups=channels*2),
            nn.Conv3d(channels * 2, channels, 1),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels * 2, 1),  # 只输出2组门控: 融合权重x2
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):

        gate_input = torch.cat([x1, self.norm(x2)], dim=1)
        gates = self.gate(gate_input)
        c = self.channels
        fusion_gate1 = gates[:, :c]
        fusion_gate2 = gates[:, c:]
        gated = torch.cat([fusion_gate1 * x1 + x1, fusion_gate2 * x2 + x2], dim=1)
        
        return gated

class Fuse3D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(Fuse3D, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv3d(channels, channels//2, kernel_size, padding=kernel_size//2, groups=channels//2),
            nn.Conv3d(channels//2, channels//2, 1),
            nn.InstanceNorm3d(channels//2),
            nn.GELU()
        )
    
    def _channel_shuffle(self, x, groups):
        batch_size, num_channels, d, h, w = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, d, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, d, h, w)
        return x
        
    def forward(self, x):
        refined = self.fuse(self._channel_shuffle(x,2))
        
        return refined


if __name__ == "__main__":
    batch_size, channels, depth, height, width = 4, 128, 24, 24, 24
    x1 = torch.randn(batch_size, channels, depth, height, width)  # 局部特征
    x2 = torch.randn(batch_size, channels, depth, height, width)  # 全局特征
    
    fusion_module = LocalGlobalFusion3D(channels=channels)
    fused_features = fusion_module(x1, x2)
    
    print(f"输入形状: x1{x1.shape}, x2{x2.shape}")
    print(f"融合输出形状: {fused_features.shape}")
    
    total_params = sum(p.numel() for p in fusion_module.parameters())*8
    print(f"融合模块参数量: {total_params / 1e6:.2f}M")  
  