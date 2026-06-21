import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import sys
sys.path.append("/root/data-tmp/MIS3D/EffiDec3D/")
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath
from mamba_ssm import Mamba  # 需要安装mamba_ssm包
from thop import profile

from networks.ours.fusionmodule import LiteAdaptiveFusion3D, StableDualAttentionFusion, MultiScaleAttention, HierarchicalMSA, LocalGlobalFusion3D
# from fusionmodule import LiteAdaptiveFusion3D, StableDualAttentionFusion
# from Adascaning import LearnableScanner3D_Medical


class Light3DStructureAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        
        # 1. 三轴并行重要性检测
        # 每个轴向都有两种感受野的卷积
        self.d_axis_convs = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3,1,1), padding=(3//2,0,0), bias=False),
            nn.GELU(),
            nn.Conv3d(8, 1, kernel_size=(3,1,1), padding=(3//2,0,0), bias=False),
            nn.GELU(),
            )
            
        
        self.h_axis_convs = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(1,3,1), padding=(0,3//2,0), bias=False),
            nn.GELU(),
            nn.Conv3d(8, 1, kernel_size=(1,3,1), padding=(0,3//2,0), bias=False),
            nn.GELU(),
            )


        self.w_axis_convs = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(1,1,3), padding=(0,0,3//2), bias=False),
            nn.GELU(),
            nn.Conv3d(8, 1, kernel_size=(1,1,3), padding=(0,0,3//2), bias=False),
            nn.GELU(),
            )

        
        # 2. 三轴权重融合
        # self.axis_weights = nn.Parameter(torch.ones(3) / 3)  # D, H, W 轴的权重
        
        # 3. 局部连续性检测
        self.continuity_conv = nn.Sequential(
            nn.InstanceNorm3d(1),
            nn.Conv3d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv3d(8, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid())
        
        # 4. 结构感知的通道权重
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 5. 全局融合权重
        # self.alpha = nn.Parameter(torch.tensor(0.5))  # 轴向重要性权重
        # self.beta = nn.Parameter(torch.tensor(0.5))   # 连续性权重
        
        self.fusion_conv = nn.Sequential(
            nn.InstanceNorm3d(3),
            nn.Conv3d(3, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid())
        
    def forward(self, x):
        b, c, d, h, w = x.shape
        
        # A. 通道注意力
        ca = self.channel_attention(x)  # [B,C,1,1,1]
        
        # B. 三维结构注意力
        # 使用通道注意力加权的特征来计算空间重要性
        x_ca = x * ca + x
        x_avg = torch.mean(x_ca, dim=1, keepdim=True)  # [B,1,D,H,W]
        
        # B1. 三轴并行重要性检测
        # 同时计算三个轴向的重要性
        d_maps = self.d_axis_convs(x_avg)
        h_maps = self.h_axis_convs(x_avg)
        w_maps = self.w_axis_convs(x_avg)
        axis_weights = torch.concat([d_maps, h_maps, w_maps], axis=1)
        axis_weights = self.fusion_conv(axis_weights)
        x_avg_axis =axis_weights * x_avg+ x_avg
        # B2. 局部连续性检测
        continuity = self.continuity_conv(x_avg_axis)  # [B,1,D,H,W]
        # print(axis_weights.shape, continuity.shape)
        # C. 注意力融合 - 直接返回空间注意力分数
        # spatial_attention = torch.sigmoid(
        #     self.alpha * axis_weights + self.beta * continuity
        # )
        
        # 返回空间注意力分数 [B, 1, D, H, W]
        return continuity*x_ca+x_ca

class LearnableScanner3D_Medical_ori(nn.Module):
    """
    使用Light3DStructureAttention的轻量化特征扫描器
    """
    def __init__(self, in_dim=16, reduction_ratio=8):
        super().__init__()
        self.in_dim = in_dim
        
        # 使用新的轻量3D结构注意力
        self.structure_attention = Light3DStructureAttention(in_dim, reduction_ratio)
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        L = D * H * W  # 总空间位置数
        
        # 1. 获取空间注意力分数 [B, 1, D, H, W]
        spatial_attention = self.structure_attention(x)
        
        # 2. 直接使用空间注意力分数进行排序扫描
        scores_flat = spatial_attention.view(B, -1)  # [B, L]
        
        # 按分数降序排列，得到扫描顺序（重要性高的优先）
        perm_idx = torch.argsort(scores_flat, dim=-1, descending=True)  # [B, L]
        
        # 3. 3D特征重排
        x_flat = x.view(B, C, -1)  # [B, C, L]
        x_scan = x_flat.gather(2, perm_idx.unsqueeze(1).expand(-1, C, -1))  # [B, C, L]
        
        # 4. 逆序索引（用于恢复原始顺序）
        inv_perm = torch.argsort(perm_idx, dim=-1)  # [B, L]
        
        return x_scan, inv_perm

    def inverse(self, x_scan, inv_perm, original_shape=None):
        
        B, C, L = x_scan.shape
        
        # 使用逆序索引恢复原始顺序
        x_restored = x_scan.gather(2, inv_perm.unsqueeze(1).expand(-1, C, -1))  # [B, C, L]
        return x_restored

class LearnableScanner3D_Medical(nn.Module):
    def __init__(self, in_dim=16, reduction_ratio=8):
        super().__init__()
        self.structure_attention = Light3DStructureAttention(in_dim, reduction_ratio)

    def forward(self, x):
        B, C, D, H, W = x.shape
        L = D * H * W

        a = self.structure_attention(x)                 # [B,1,D,H,W]
        w = torch.sigmoid(a).reshape(B, C, L)           # [B,1,L] 作为门控，范围稳定

        x_flat = x.reshape(B, C, L)                     # [B,C,L]
        x_scan = x_flat * (1.0 + w)                     # [B,C,L] 学习到的“重要性增益”
        inv_perm = None
        return x_scan





class DWConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)

class adamamba(nn.Module):
    """深度可分离卷积"""
    def __init__(self, dim=64):
        super(adamamba, self).__init__()
        self.scanner1 = LearnableScanner3D_Medical(in_dim=dim, reduction_ratio=8)
        self.mambalayer = Mamba(
                d_model=dim,
                d_state=16,
                d_conv=4,
                expand=2
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # print('0',x.shape)
        x_scan1 = self.scanner1(x)
        x_scan1 = self.mambalayer(x_scan1.permute(0, 2, 1))
        x_scan1 = self.norm(x_scan1)

        # x_recovered1 = self.scanner1.inverse(x_scan1.permute(0, 2, 1), inv_perm1)
        
        return x_scan1

class MambaBlock(nn.Module):
    """Mamba块，与卷积并行"""
    def __init__(self, dim, depth=2, drop_path=0., mlp_ratio=4., 
                 conv_type='depthwise', norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # Mamba分支
        # self.mamba_layers = nn.ModuleList([
        #     Mamba(
        #         d_model=dim,
        #         d_state=16,
        #         d_conv=4,
        #         expand=2
        #     ) for _ in range(depth)
        # ])
        
        self.mamba_layers = nn.ModuleList([
            # Mamba(
            #     d_model=dim,
            #     d_state=16,
            #     d_conv=4,
            #     expand=2
            # ) 
            adamamba(dim=dim)
            for _ in range(depth)
        ])
        
        # 卷积分支
        # if conv_type == 'depthwise':
        self.conv_layers = nn.ModuleList([
                nn.Sequential(
                    DWConv(dim),
                    nn.Conv3d(dim, dim, 1),  # 逐点卷积
                    nn.InstanceNorm3d(dim),
                    nn.GELU(),
                ) for _ in range(depth)
            ])
        # else:
        #     self.conv_layers = nn.ModuleList([
        #         nn.Sequential(
        #             nn.Conv3d(dim, dim, 3, 1, 1, groups=dim//4),
        #             nn.Conv3d(dim, dim, 1),  # 逐点卷积
        #             nn.GELU(),
        #         ) for _ in range(depth)
        #     ])
        
        # 特征融合
        # self.fusion_module = LocalGlobalFusion3D(channels=channels)
        self.fusion_module = nn.ModuleList([
            LocalGlobalFusion3D(channels=dim) for _ in range(depth)
        ])
        # self.fusion_layers = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv3d(dim * 2, dim, 1),
        #         nn.InstanceNorm3d(dim),
        #         nn.GELU(),
        #         # nn.LayerNorm(dim)
        #     ) for _ in range(depth)
        # ])
        
        # 归一化和drop path
        self.norms = nn.ModuleList([norm_layer(dim) for _ in range(depth)])
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # MLP
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = nn.Sequential(
        #     nn.Linear(dim, mlp_hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(mlp_hidden_dim, dim),
        #     nn.Dropout(0.1)
        # )
        # self.mlp_norm = norm_layer(dim)

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
        B, C, D, H, W = x.shape
        # print(x.shape)
        # 将3D特征转换为序列格式 (B, L, C)
        x_sequence = rearrange(x, 'b c d h w -> b (d h w) c')
        # x_sequence = x
        for i in range(self.depth):
            # print(i)
            # Mamba分支
            x_mamba = self.mamba_layers[i](rearrange(x_sequence, 'b (d h w) c -> b c d h w', d=D, h=H, w=W))
            # print(x_mamba.shape)
            # x_mamba = self.norms[i * 2](x_mamba)
            # print(i)
            # 卷积分支
            x_conv = rearrange(x_sequence, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
            x_conv = self.conv_layers[i](x_conv)
            x_conv = rearrange(x_conv, 'b c d h w -> b (d h w) c')
            # x_conv = self.norms[i](x_conv)
            # print(x_mamba.shape, x_conv.shape)
            # 特征融合
            
            # x_fused = self.fusion_module[i](x_conv, rearrange(x_mamba, 'b (d h w) c -> b c d h w', d=D, h=H, w=W))
            
            # x_fused = torch.cat([rearrange(x_mamba, 'b (d h w) c -> b c d h w', d=D, h=H, w=W), x_conv], dim=1)
            # # print(x_fused.shape)
            # x_fused = self._channel_shuffle(x_fused, groups=2)
            # x_fused = self.fusion_layers[i](x_fused)
            # x_fused = rearrange(x_fused, 'b c d h w -> b (d h w) c')
            
            # 残差连接
            x_sequence = x_sequence + self.drop_path(x_conv)
        
        # MLP层
        # x_sequence = x_sequence + self.drop_path(self.mlp(self.mlp_norm(x_sequence)))
        
        # 转换回3D格式
        x_out = rearrange(x_sequence, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        return x_out

class EncoderBlock(nn.Module):
    """编码器块：下采样 + Mamba并行块"""
    def __init__(self, in_channels, out_channels, depth=2, drop_path=0., 
                 downsample=True, stage=1):
        super().__init__()
        
        # 下采样
        # if downsample:
        self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, 2, 1, groups=in_channels),
                nn.Conv3d(out_channels, out_channels, 1),
                nn.InstanceNorm3d(out_channels),
                nn.GELU()
            )
        # else:
        #     self.downsample = nn.Sequential(
        #         nn.Conv3d(in_channels, out_channels, 3, 1, 1, groups=in_channels),
        #         nn.Conv3d(out_channels, out_channels, 1),
        #         nn.InstanceNorm3d(out_channels),
        #         nn.GELU()
        #     )
        
        # Mamba并行块
        self.mamba_blocks = MambaBlock(
            dim=out_channels,
            depth=depth,
            drop_path=drop_path
        )

    
    
    def forward(self, x):
        x = self.downsample(x)
        # print(x.shape)
        x = self.mamba_blocks(x)
        return x

class DecoderBlock(nn.Module):
    """解码器块：上采样 + 跳跃连接 + 卷积"""
    def __init__(self, in_channels, skip_channels, out_channels, depth=2):
        super().__init__()
        
        # 上采样
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, 2),
            nn.InstanceNorm3d(out_channels),
            nn.GELU()
        )
        
        # 跳跃连接融合
        self.skip_conv = nn.Sequential(
            nn.Conv3d(skip_channels + out_channels, out_channels, 1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU()
        )
        
        # self.fusion = LiteAdaptiveFusion3D(
        #     in_ch_e=out_channels,
        #     in_ch_d=out_channels,
        #     out_ch=out_channels,       # 该解码阶段期望的输出通道
        #     t=2,             # 通道压缩倍率，越大越轻
        #     hidden_ratio=1,  # 细化块倍率，建议 1
        #     groups=out_channels//2,
        #     upsample_mode="trilinear",
        #     use_align=True   # 自动把 enc_feat 尺寸对齐 dec_feat
        # )
        # self.att = StableDualAttentionFusion(out_channels, out_channels)
        # self.att = MultiScaleAttention(out_channels)
        self.att = HierarchicalMSA(out_channels)


        
        # 卷积块
        conv_blocks = []
        for _ in range(depth):
            conv_blocks.extend([
                nn.Conv3d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
                nn.Conv3d(out_channels, out_channels, 1),
                nn.InstanceNorm3d(out_channels),
                nn.GELU()
            ])
        self.conv_blocks = nn.Sequential(*conv_blocks)

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
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # 处理跳跃连接尺寸不匹配
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        
        # 跳跃连接融合
        # print(x.shape, skip.shape)
        x = torch.cat([x, skip], dim=1)
        x = self.skip_conv(self._channel_shuffle(x, 2))
        # sample = x
        x = self.att(x)
        # x = self.fusion(skip, x)
        
        # 卷积细化
        x = self.conv_blocks(x) + x
        return x

class MCVUNet(nn.Module):
    """基于Mamba和卷积并行的3D医学图像分割网络"""
    
    def __init__(self, in_channels=4, out_channels=4, base_dim=32, depths=[2, 2, 2, 2]):
        super().__init__()
        
        # 初始卷积 - 修改为不下采样，保持96x96x96分辨率
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_dim, 7, 1, 3),  # stride=1, 不下采样
            nn.InstanceNorm3d(base_dim),
            nn.GELU()
        )
        
        # 编码器阶段 - 调整下采样策略以保持最终输出为96x96x96
        self.encoders = nn.ModuleList()
        dims = [base_dim, base_dim*2, base_dim*4, base_dim*8, base_dim*16]
        
        # 修改下采样策略：只在特定阶段下采样
        for i in range(4):
            # 只在第1、2阶段下采样，第3、4阶段保持分辨率
            downsample = (i < 4)  # 只有前两个阶段下采样
            
            self.encoders.append(
                EncoderBlock(
                    in_channels=dims[i],
                    out_channels=dims[i+1],
                    depth=depths[i],
                    drop_path=0.1 * i,
                    downsample=downsample,
                    stage=i+1
                )
            )
        
        # 瓶颈层 - 保持分辨率
        # self.bottleneck = nn.Sequential(
        #     MambaBlock(dim=dims[4], depth=2),
        #     nn.Conv3d(dims[4], dims[4], 3, 1, 1),
        #     nn.InstanceNorm3d(dims[4]),
        #     nn.GELU()
        # )
        
        # 解码器阶段 - 对称于编码器
        self.decoders = nn.ModuleList()
        for i in range(3, -1, -1):
            # 对称于编码器的下采样策略
            self.decoders.append(
                DecoderBlock(
                    in_channels=dims[i+1],
                    skip_channels=dims[i],
                    out_channels=dims[i],
                    depth=2
                )
            )
        self.out2 = nn.Sequential(
            # nn.ConvTranspose3d(in_channels, out_channels, 2, 2),
            # nn.InstanceNorm3d(out_channels),
            # nn.GELU()
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(base_dim*2, base_dim, 3, 1, 1,groups=base_dim),
            nn.Conv3d(base_dim, out_channels, 1)
        )
        self.out3 = nn.Sequential(
            # nn.ConvTranspose3d(in_channels, out_channels, 2, 2),
            # nn.InstanceNorm3d(out_channels),
            # nn.GELU()
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True),
            nn.Conv3d(base_dim*4, base_dim, 3, 1, 1,groups=base_dim),
            nn.Conv3d(base_dim, out_channels, 1)
        )
        self.out4 = nn.Sequential(
            # nn.ConvTranspose3d(in_channels, out_channels, 2, 2),
            # nn.InstanceNorm3d(out_channels),
            # nn.GELU()
            nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True),
            nn.Conv3d(base_dim*8, base_dim, 3, 1, 1,groups=base_dim),
            nn.Conv3d(base_dim, out_channels, 1)
        )
        
        # 最终输出层 - 直接输出，不需要上采样
        self.output_conv = nn.Sequential(
            nn.Conv3d(base_dim, base_dim//2, 3, 1, 1,groups=base_dim//2),
            nn.Conv3d(base_dim//2, base_dim//2, 1),  # 逐点卷积
            nn.InstanceNorm3d(base_dim//2),
            nn.GELU(),
            nn.Conv3d(base_dim//2, out_channels, 1)
            
        )
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 记录输入尺寸
        input_size = x.shape[2:]
        
        # 初始卷积 - 保持分辨率
        x0 = self.stem(x)  # 保持96x96x96分辨率
        
        # 编码器路径
        skips = [x0]
        x = x0
        
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < 3:  # 保存跳跃连接（除了最后一个）
                skips.append(x)
        
        # 瓶颈层
        
        outs = []
        # 解码器路径
        for i, decoder in enumerate(self.decoders):
            skip = skips[3-i]  # 从后往前取跳跃连接
            x = decoder(x, skip)
            if i < 3:  # 保存输出（除了最后一个）
                outs.append(x)
        
        # 输出层
        x = self.output_conv(x)
        out2 = self.out2(outs[2])
        out3 = self.out3(outs[1])
        out4 = self.out4(outs[0])
        # 确保输出尺寸与输入一致
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='trilinear', align_corners=False)
        
        return [x, out2, out3, out4]

# 测试代码

    

if __name__ == "__main__":
    # 检查GPU是否可用
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"使用设备: {device}")
    
    # 创建模型并移动到GPU
    model = MCVUNet(in_channels=1, out_channels=9, base_dim=32)
    model = model.cuda()
    
    # 测试输入
    x = torch.randn(1, 1, 96, 96, 96).cuda()
    
    # 前向传播
    with torch.no_grad():
        output = model(x)[0]
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 检查设备一致性
    # print(f"输入设备: {x.device}")
    # print(f"输出设备: {output.device}")
    # print(f"模型设备: {next(model.parameters()).device}")
    
    # 验证输入输出尺寸是否一致
    if x.shape == output.shape:
        print("✓ 输入输出尺寸匹配!")
    else:
        print("✗ 输入输出尺寸不匹配!")
    
    # 使用thop计算FLOPs和参数量
    from thop import profile
    
    def count_flops_and_params(model, input_shape):
        model.eval()
        input_tensor = torch.randn(*input_shape).cuda()
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        return flops, params
    
    # 计算FLOPs和参数量
    input_shape = (1, 1, 96, 96, 96)
    total_flops, total_params = count_flops_and_params(model, input_shape)
    print(f"模型FLOPs: {total_flops / 1e9:.2f} GFLOPs")
    print(f"thop计算参数量: {total_params / 1e6:.2f}M")
    
    # 验证参数量计算是否一致
    manual_params = sum(p.numel() for p in model.parameters())
    if abs(total_params - manual_params) < 100:
        print("✓ 参数量计算一致")
    else:
        print(f"⚠ 参数量计算有差异: 手动{manual_params}, thop{total_params}")
    
    
    
    