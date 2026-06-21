import torch
import torch.nn as nn
import torch.nn.functional as F


class PW_GN_Act3D(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8, act=True):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        # self.gn = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.norm = nn.InstanceNorm3d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparable3D_Lite(nn.Module):
    """ 1×1×1 -> DW 3×3×3 -> 1×1×1，尽量小的细化块 """
    def __init__(self, ch, hidden_ratio=1, groups=8):
        super().__init__()
        hidden = max(ch, 8) * hidden_ratio
        # self.pw1 = PW_GN_Act3D(ch, hidden, groups=groups, act=True)
        self.dw  = nn.Conv3d(ch, hidden, kernel_size=3, padding=1, groups=hidden, bias=False)
        # self.dw_gn = nn.GroupNorm(num_groups=min(groups, hidden), num_channels=hidden)
        self.pw2 = PW_GN_Act3D(hidden, ch, groups=groups, act=True)
    def forward(self, x):
        # x = self.pw1(x)
        # x = self.dw_gn(self.dw(x))
        x = self.dw(x)
        # x = F.gelu(x)
        x = self.pw2(x)
        return x


class LiteAdaptiveFusion3D(nn.Module):
    """
    轻量化自适应拼接融合模块
    输入:
        E: 编码器特征 (B, C_e, D, H, W)
        D: 解码器特征 (B, C_d, D, H, W)
    输出:
        out: 融合特征 (B, out_ch, D, H, W)
    关键超参:
        t: 通道压缩倍率，越大越轻
        hidden_ratio: 细化块的隐层倍率，建议 1
    """
    def __init__(
        self,
        in_ch_e: int,
        in_ch_d: int,
        out_ch: int,
        t: int = 2,
        groups: int = 8,
        hidden_ratio: int = 1,
        upsample_mode: str = "trilinear",
        use_align: bool = True
    ):
        super().__init__()
        self.use_align = use_align
        self.upsample_mode = upsample_mode

        # 通道瓶颈
        mid_ch = max(out_ch // t, 16)
        self.proj_e = PW_GN_Act3D(in_ch_e, mid_ch, groups=groups, act=True)
        self.proj_d = PW_GN_Act3D(in_ch_d, mid_ch, groups=groups, act=True)

        # 共享 Squeeze MLP 产生通道 logits（对 E 与 D 共享权重）
        red = max(1, mid_ch // 8)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.ch_fc1 = nn.Conv3d(mid_ch, red, kernel_size=1, bias=True)
        self.ch_fc2 = nn.Conv3d(red, mid_ch, kernel_size=1, bias=True)

        # 极简空间门控：先对拼接特征按通道平均，再 DW 生成 2 源 logits
        self.spatial_dw = nn.Conv3d(1, 1, kernel_size=3, padding=1, groups=1, bias=False)
        self.spatial_pw = nn.Conv3d(1, 2, kernel_size=1, bias=True)

        # 轻量细化与输出
        self.refine = DepthwiseSeparable3D_Lite(mid_ch, hidden_ratio=hidden_ratio, groups=groups)
        self.out = PW_GN_Act3D(mid_ch, out_ch, groups=groups, act=True)

        # 残差对齐
        self.res_proj = None
        if in_ch_d != out_ch:
            self.res_proj = nn.Conv3d(in_ch_d, out_ch, kernel_size=1, bias=False)

    def _align(self, E, D):
        if self.use_align and (E.shape[2:] != D.shape[2:]):
            E = F.interpolate(E, size=D.shape[2:], mode=self.upsample_mode, align_corners=False)
        return E

    def _channel_logits(self, X):
        # 共享 MLP 对输入 X 产生通道 logits
        z = self.gap(X)              # B, C, 1, 1, 1
        z = F.gelu(self.ch_fc1(z))   # B, red, 1, 1, 1
        z = self.ch_fc2(z)           # B, C, 1, 1, 1
        return z

    def forward(self, E, D):
        # E = self._align(E, D)
        E = self.proj_e(E)    # B, C, D, H, W
        Dp = self.proj_d(D)   # B, C, D, H, W
        C = E.shape[1]

        # 通道门控（对 E 与 D 共享参数，再在来源维度 softmax）
        logit_c_e = self._channel_logits(E)      # B, C, 1, 1, 1
        logit_c_d = self._channel_logits(Dp)     # B, C, 1, 1, 1
        w_c = torch.stack([logit_c_e, logit_c_d], dim=1)  # B, 2, C, 1, 1, 1
        w_c = F.softmax(w_c, dim=1)
        w_c_e, w_c_d = w_c[:, 0], w_c[:, 1]      # B, C, 1, 1, 1

        # 空间门控（极简 DW + PW）
        X_cat = torch.cat([E, Dp], dim=1)        # B, 2C, D, H, W
        X_mean = X_cat.mean(dim=1, keepdim=True) # B, 1, D, H, W
        s = self.spatial_pw(self.spatial_dw(X_mean))  # B, 2, D, H, W
        w_s = F.softmax(s, dim=1)
        w_s_e, w_s_d = w_s[:, 0:1], w_s[:, 1:2]  # B, 1, D, H, W

        # 双门控融合并归一化
        ge = w_c_e * w_s_e
        gd = w_c_d * w_s_d
        eps = 1e-6
        norm = ge + gd + eps
        ge, gd = ge / norm, gd / norm

        F_fuse = ge * E + gd * Dp               # B, C, D, H, W

        # 轻量细化与输出
        F_ref = self.refine(F_fuse)             # B, C, D, H, W
        out = self.out(F_ref)                   # B, out_ch, D, H, W

        # 残差
        res = D if self.res_proj is None else self.res_proj(D)
        out = out + res
        return out
    
    
    
class StableDualAttentionFusion(nn.Module):
    def __init__(self, skip_channels, out_channels, reduction=16):
        super().__init__()
        
        # 通道注意力 - 使用稳定的SE模块变体
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_channels, out_channels // reduction, 1),
            nn.ReLU(inplace=True),  # 使用ReLU更稳定
            nn.Conv3d(out_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )
        
        # 稳定的轴向空间注意力
        self.spatial_attention = StableAxialSpatialAttention(out_channels)
        
        self.norm = nn.BatchNorm3d(out_channels)
        # 基础融合卷积 + 稳定的归一化
        # self.skip_conv = nn.Sequential(
        #     nn.Conv3d(skip_channels + out_channels, out_channels, 1),
        #     nn.BatchNorm3d(out_channels),  # BatchNorm比InstanceNorm更稳定
        #     nn.ReLU(inplace=True)
        # )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # 残差连接保留原始信息
        identity = x
        
        # 通道注意力
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights + x
        
        # 空间注意力
        x_spatial = self.spatial_attention(x_channel)
        
        # 渐进式融合 + 残差连接
        # x_enhanced = 0.7 * x_spatial + 0.3 * identity
        
        # 最终融合
        # fused = torch.cat([x_enhanced, skip], dim=1)
        # return self.skip_conv(fused)
        return self.norm(x_spatial)+identity

class StableAxialSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        # 使用更稳定的设计
        self.depth_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(channels, 1, 1),
            nn.Sigmoid()
        )
        
        self.height_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, None, 1)),
            nn.Conv3d(channels, 1, 1),
            nn.Sigmoid()
        )
        
        self.width_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, None)),
            nn.Conv3d(channels, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 分别计算三个轴向的注意力
        depth_att = self.depth_attention(x)
        height_att = self.height_attention(x)
        width_att = self.width_attention(x)
        
        # 渐进式组合，避免极端值
        spatial_att = (depth_att + height_att + width_att) / 3
        
        # 使用更温和的加权
        return x * (1 + spatial_att)  # 确保输出不会太小


import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups):
    """Channel Shuffle操作，促进不同分组间的信息交流"""
    batch_size, num_channels, depth, height, width = x.size()
    channels_per_group = num_channels // groups

    # 重塑为 [batch_size, groups, channels_per_group, depth, height, width]
    x = x.view(batch_size, groups, channels_per_group, depth, height, width)

    # 转置维度 [batch_size, channels_per_group, groups, depth, height, width]
    x = torch.transpose(x, 1, 2).contiguous()

    # 重塑回原始形状
    x = x.view(batch_size, -1, depth, height, width)

    return x


class MultiScaleAttention(nn.Module):
    """
    串行轻量化多尺度通道空间注意力机制
    结构：输入 -> 通道注意力 -> 多尺度空间注意力 -> 输出
    """

    def __init__(self, channels, reduction=16, kernel_sizes=[3, 3, 5], use_residual=True):
        super(MultiScaleAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.kernel_sizes = kernel_sizes
        self.use_residual = use_residual
        self.dilations = [1, 2, 1]
        # ==================== 通道注意力 ====================
        self.channel_attention = nn.Sequential(
            # 全局平均池化
            nn.AdaptiveAvgPool3d(1),
            # 第一个全连接层，降维
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            # 第二个全连接层，恢复维度
            nn.Conv3d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()  # 输出通道权重 [0,1]
        )

        # ==================== 多尺度空间注意力 ====================
        self.spatial_branches = nn.ModuleList()
        for k_size, dilation in zip(self.kernel_sizes, self.dilations):
            # 每个分支使用深度可分离卷积
            branch = nn.Sequential(
                # 深度卷积 - 提取空间特征，保持通道数不变
                nn.Conv3d(1, 1, kernel_size=k_size, padding=(k_size+dilation-1) // 2, dilation=dilation, groups=1, bias=False),
                # nn.InstanceNorm3d(1),
                # 点卷积 - 升维到8
                nn.Conv3d(1, 8, kernel_size=1, bias=False),
                nn.InstanceNorm3d(8),
                nn.GELU()
            )
            self.spatial_branches.append(branch)

        # 计算总通道数（每个分支输出8通道，拼接后）
        total_channels = 8 * len(kernel_sizes)

        # 拼接后的融合卷积
        self.spatial_fusion = nn.Sequential(
            # 点卷积，将拼接后的特征融合并降维到1通道
            nn.Conv3d(total_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()  # 输出空间注意力权重 [0,1]
        )


    def forward(self, x):
        residual = x

        # ========== 第一步：通道注意力 ==========
        channel_weights = self.channel_attention(x)
        channel_refined = x * channel_weights

        # ========== 第二步：多尺度空间注意力 ==========
        # 将通道细化后的特征转换为空间注意力输入
        spatial_input = torch.mean(channel_refined, dim=1, keepdim=True)  # [B, 1, D, H, W]

        # 多尺度空间注意力并行计算
        spatial_features = []
        for branch in self.spatial_branches:
            # 每个分支处理并输出8通道特征
            branch_output = branch(spatial_input)  # [B, 8, D, H, W]
            spatial_features.append(branch_output)

        # 拼接所有分支的特征
        concatenated = torch.cat(spatial_features, dim=1)  # [B, 8*len(kernel_sizes), D, H, W]

        # 应用Channel Shuffle促进信息交流
        shuffled = channel_shuffle(concatenated, groups=len(self.kernel_sizes))

        # 通过点卷积融合并得到空间注意力图
        spatial_attention = self.spatial_fusion(shuffled)  # [B, 1, D, H, W]

        # 应用空间注意力到所有通道
        output = channel_refined * spatial_attention + residual

        return output


import torch
import torch.nn as nn

class HierarchicalMSA(nn.Module):
    """
    串行轻量化多尺度通道空间注意力机制
    结构：输入 -> 通道注意力 -> 多尺度空间注意力 -> 输出
    """
    def __init__(self, channels, reduction=8, use_residual=True, CA = True, SA = True):
        super(HierarchicalMSA, self).__init__()
        self.channels = channels
        self.reduction = reduction
        # self.kernel_sizes = kernel_sizes
        self.use_residual = use_residual
        self.dilations = [1, 2, 1]
        self.ca = CA
        self.sa = SA
        # ==================== 通道注意力 ====================
        self.channel_attention = nn.Sequential(
            # 全局平均池化
            nn.AdaptiveAvgPool3d(1),
            # 第一个全连接层，降维
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            # 第二个全连接层，恢复维度
            nn.Conv3d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()  # 输出通道权重 [0,1]
        )

        self.branch1 = nn.Sequential(
            # 深度卷积 - 提取空间特征，保持通道数不变
            nn.Conv3d(1, 4, kernel_size=3, padding=1, dilation=1, groups=1, bias=False),
            nn.Conv3d(4, 1, kernel_size=1, bias=False),
            nn.InstanceNorm3d(4),
            nn.GELU())
        self.branch2 = nn.Sequential(
            # 深度卷积 - 提取空间特征，保持通道数不变
            nn.Conv3d(2, 8, kernel_size=3, padding=2, dilation=2, groups=2, bias=False),
            nn.Conv3d(8, 2, kernel_size=1, bias=False),
            nn.InstanceNorm3d(8),
            nn.GELU())
        self.branch3 = nn.Sequential(
            # 深度卷积 - 提取空间特征，保持通道数不变
            nn.Conv3d(3, 12, kernel_size=5, padding=2, dilation=1, groups=3, bias=False),
            nn.Conv3d(12, 3, kernel_size=1, bias=False),
            nn.InstanceNorm3d(12),
            nn.GELU())

        # 拼接后的融合卷积
        self.spatial_fusion = nn.Sequential(
            # 点卷积，将拼接后的特征融合并降维到1通道
            nn.Conv3d(4, 4, kernel_size=3, groups=4, padding=1, bias=False),
            nn.Conv3d(4, 1, kernel_size=1, bias=False),
            nn.Sigmoid()  # 输出空间注意力权重 [0,1]
        )


    def forward1(self, x):
        residual = x

        # ========== 第一步：通道注意力 ==========
        channel_weights = self.channel_attention(x)
        channel_refined = x * channel_weights + x

        # ========== 第二步：多尺度空间注意力 ==========
        # 将通道细化后的特征转换为空间注意力输入
        spatial_input = torch.mean(channel_refined, dim=1, keepdim=True)  # [B, 1, D, H, W]

        # 多尺度空间注意力层级计算
        out1 = torch.cat([self.branch1(spatial_input), spatial_input], dim=1)
        out2 = torch.cat([self.branch2(out1), spatial_input], dim=1)
        out3 = torch.cat([self.branch3(out2), spatial_input], dim=1)

        # 通过点卷积融合并得到空间注意力图
        spatial_attention = self.spatial_fusion(out3)  # [B, 1, D, H, W]

        # 应用空间注意力到所有通道
        output = channel_refined * spatial_attention + channel_refined

        return output
    
    def forward2(self, x):
        residual = x

        # ========== 第一步：通道注意力 ==========
        channel_weights = self.channel_attention(x)
        output = x * channel_weights + x

        # ========== 第二步：多尺度空间注意力 ==========
        # 将通道细化后的特征转换为空间注意力输入
        # spatial_input = torch.mean(channel_refined, dim=1, keepdim=True)  # [B, 1, D, H, W]

        # # 多尺度空间注意力层级计算
        # out1 = torch.cat([self.branch1(spatial_input), spatial_input], dim=1)
        # out2 = torch.cat([self.branch2(out1), spatial_input], dim=1)
        # out3 = torch.cat([self.branch3(out2), spatial_input], dim=1)

        # # 通过点卷积融合并得到空间注意力图
        # spatial_attention = self.spatial_fusion(out3)  # [B, 1, D, H, W]

        # # 应用空间注意力到所有通道
        # output = channel_refined * spatial_attention + channel_refined

        return output
    
    def forward3(self, x):
        residual = x

        # ========== 第一步：通道注意力 ==========
        # channel_weights = self.channel_attention(x)
        # channel_refined = x * channel_weights + x

        # ========== 第二步：多尺度空间注意力 ==========
        # 将通道细化后的特征转换为空间注意力输入
        spatial_input = torch.mean(x, dim=1, keepdim=True)  # [B, 1, D, H, W]

        # 多尺度空间注意力层级计算
        out1 = torch.cat([self.branch1(spatial_input), spatial_input], dim=1)
        out2 = torch.cat([self.branch2(out1), spatial_input], dim=1)
        out3 = torch.cat([self.branch3(out2), spatial_input], dim=1)

        # 通过点卷积融合并得到空间注意力图
        spatial_attention = self.spatial_fusion(out3)  # [B, 1, D, H, W]

        # 应用空间注意力到所有通道
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

  
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalGlobalFusion3D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=3):
        super(LocalGlobalFusion3D, self).__init__()
        self.channels = channels
        
        # 简化的门控融合模块
        self.gate_fusion = SimplifiedGatedFusion3D(channels, kernel_size)
        
        # 特征增强模块
        self.enhancement = FeatureEnhancement3D(channels*2, kernel_size)
        
        # 输出卷积
        # self.output_conv = nn.Conv3d(channels, channels, 1)
        
        # 残差连接
        # self.residual_conv = nn.Conv3d(channels, channels, 1)
        
    def forward(self, x1, x2):
        """
        x1: 局部特征 [B, C, D, H, W]
        x2: 全局特征 [B, C, D, H, W]
        """
        # 保存输入用于残差连接
        # identity = x1
        
        # 门控融合
        fused = self.gate_fusion(x1, x2)
        
        # 特征增强
        enhanced = self.enhancement(fused)
        
        # 输出卷积 + 残差连接
        # output = self.output_conv(enhanced)
        # residual = self.residual_conv(identity)
        
        return enhanced

class SimplifiedGatedFusion3D(nn.Module):
    """简化的3D门控融合模块"""
    def __init__(self, channels, kernel_size=3):
        super(SimplifiedGatedFusion3D, self).__init__()
        padding = kernel_size // 2
        self.channels = channels
        # 特征变换 - 3D深度可分离卷积
        # self.local_transform = nn.Sequential(
        #     nn.Conv3d(channels, channels, kernel_size, padding=padding, groups=channels),
        #     nn.Conv3d(channels, channels, 1),
        #     nn.InstanceNorm3d(channels)
        # )
        
        # self.global_transform = nn.Sequential(
        #     nn.InstanceNorm3d(channels),
        #     nn.Conv3d(channels, channels, kernel_size, padding=padding, groups=channels),
        #     nn.Conv3d(channels, channels, 1),
        #     nn.InstanceNorm3d(channels)
        # )
        self.norm = nn.InstanceNorm3d(channels)
        # 简化的门控网络 - 只生成2组融合权重
        self.simple_gate = nn.Sequential(
            nn.Conv3d(channels * 2, channels * 2, kernel_size, padding=padding, groups=channels*2),
            nn.Conv3d(channels * 2, channels, 1),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels * 2, 1),  # 只输出2组门控: 融合权重x2
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        # 特征变换
        # x1_trans = F.gelu(self.local_transform(x1))
        # x2_trans = F.gelu(self.global_transform(x2))
        
        # 简化门控生成
        gate_input = torch.cat([x1, self.norm(x2)], dim=1)
        gates = self.simple_gate(gate_input)  # [B, C*2, D, H, W]
        
        # 分离门控信号
        c = self.channels
        fusion_gate1 = gates[:, :c]      # 局部特征融合权重
        fusion_gate2 = gates[:, c:]      # 全局特征融合权重
        
        # 直接融合
        fused = torch.cat([fusion_gate1 * x1 + x1, fusion_gate2 * x2 + x2], dim=1) #0
        # fused = torch.cat([fusion_gate1 * x1, fusion_gate2 * x2], dim=1) #1
        
        return fused

class FeatureEnhancement3D(nn.Module):
    """3D特征增强模块"""
    def __init__(self, channels, kernel_size=3):
        super(FeatureEnhancement3D, self).__init__()
        
        # 使用门控卷积进行特征增强
        # self.gated_conv1 = DepthwiseGatedConv3D(channels, channels//2, kernel_size)
        # self.gated_conv2 = DepthwiseGatedConv3D(channels, channels, kernel_size)
        
        # 特征提炼
        self.refinement = nn.Sequential(
            nn.Conv3d(channels, channels//2, kernel_size, padding=kernel_size//2, groups=channels//2),
            nn.Conv3d(channels//2, channels//2, 1),
            nn.InstanceNorm3d(channels//2),
            nn.GELU()
        )
    
    def _channel_shuffle(self, x, groups):
        batch_size, num_channels, d, h, w = x.size()
        channels_per_group = num_channels // groups
    
        # 重塑为 (batch_size, groups, channels_per_group, d, h, w)
        x = x.view(batch_size, groups, channels_per_group, d, h, w)
    
        # 转置维度 (batch_size, groups, channels_per_group, d, h, w) -> 
        # (batch_size, channels_per_group, groups, d, h, w)
        x = x.transpose(1, 2).contiguous()
    
        # 重塑回原始形状
        x = x.view(batch_size, -1, d, h, w)
        return x
        
    def forward(self, x):
        # 第一层门控卷积
        # out = self.gated_conv1(x)# + x
        
        # 第二层门控卷积 + 残差连接
        # out2 = self.gated_conv2(out1)
        
        # 特征提炼
        refined = self.refinement(self._channel_shuffle(x,2))
        
        return refined  # 残差连接

class DepthwiseGatedConv3D(nn.Module):
    """3D深度可分离门控卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseGatedConv3D, self).__init__()
        padding = kernel_size // 2
        
        # 特征变换路径 - 3D深度可分离卷积
        self.feature_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels),
            nn.Conv3d(in_channels, out_channels, 1),
            nn.InstanceNorm3d(out_channels)
        )
        
        # 门控路径 - 3D深度可分离卷积
        self.gate_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels),
            nn.Conv3d(in_channels, out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.feature_conv(x)
        gate = self.gate_conv(x)
        return features * gate

# 调用方式
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
  