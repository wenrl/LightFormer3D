import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath
from mamba_ssm import Mamba
from thop import profile
from networks.ours.fusionmodule import MSA, LocalGlobalFusion3D



class Light3DStructureAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
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

        self.continuity_conv = nn.Sequential(
            nn.InstanceNorm3d(1),
            nn.Conv3d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv3d(8, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid())

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        
        self.fusion_conv = nn.Sequential(
            nn.InstanceNorm3d(3),
            nn.Conv3d(3, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid())
        
    def forward(self, x):

        ca = self.channel_attention(x)
        x_ca = x * ca + x
        x_avg = torch.mean(x_ca, dim=1, keepdim=True)

        d_maps = self.d_axis_convs(x_avg)
        h_maps = self.h_axis_convs(x_avg)
        w_maps = self.w_axis_convs(x_avg)
        axis_weights = torch.concat([d_maps, h_maps, w_maps], axis=1)
        axis_weights = self.fusion_conv(axis_weights)
        x_avg_axis =axis_weights * x_avg+ x_avg
        continuity = self.continuity_conv(x_avg_axis)  # [B,1,D,H,W]

        return continuity*x_ca+x_ca

class GAC(nn.Module):
    def __init__(self, in_dim=16, reduction_ratio=8):
        super().__init__()
        self.structure_attention = Light3DStructureAttention(in_dim, reduction_ratio)

    def forward(self, x):
        B, C, D, H, W = x.shape
        L = D * H * W

        a = self.structure_attention(x)
        w = torch.sigmoid(a).reshape(B, C, L)

        x_flat = x.reshape(B, C, L)
        x_scan = x_flat * (1.0 + w)

        return x_scan

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)

class gacmamba(nn.Module):

    def __init__(self, dim=64):
        super(gacmamba, self).__init__()
        self.att = GAC(in_dim=dim, reduction_ratio=8)
        self.mambalayer = Mamba(
                d_model=dim,
                d_state=16,
                d_conv=4,
                expand=2
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x_scan = self.att(x)
        x_scan = self.mambalayer(x_scan.permute(0, 2, 1))
        x_scan = self.norm(x_scan)
        
        return x_scan

class MambaBlock(nn.Module):

    def __init__(self, dim, depth=2, drop_path=0., mlp_ratio=4., 
                 conv_type='depthwise', norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mamba_layers = nn.ModuleList([
            gacmamba(dim=dim)
            for _ in range(depth)
        ])

        self.conv_layers = nn.ModuleList([
                nn.Sequential(
                    DWConv(dim),
                    nn.Conv3d(dim, dim, 1),  # 逐点卷积
                    nn.InstanceNorm3d(dim),
                    nn.GELU(),
                ) for _ in range(depth)
            ])

        self.fusion_module = nn.ModuleList([
            LocalGlobalFusion3D(channels=dim) for _ in range(depth)
        ])

        self.norms = nn.ModuleList([norm_layer(dim) for _ in range(depth)])
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _channel_shuffle(self, x, groups):
        batch_size, num_channels, d, h, w = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, d, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, d, h, w)
        return x
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        x_sequence = rearrange(x, 'b c d h w -> b (d h w) c')
        for i in range(self.depth):
            x_mamba = self.mamba_layers[i](rearrange(x_sequence, 'b (d h w) c -> b c d h w', d=D, h=H, w=W))
            x_conv = rearrange(x_sequence, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
            x_conv = self.conv_layers[i](x_conv)
            x_fused = self.fusion_module[i](x_conv, rearrange(x_mamba, 'b (d h w) c -> b c d h w', d=D, h=H, w=W))
            x_fused = rearrange(x_fused, 'b c d h w -> b (d h w) c')
            x_sequence = x_sequence + self.drop_path(x_fused)
        x_out = rearrange(x_sequence, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        return x_out

class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, depth=2, drop_path=0., 
                 downsample=True, stage=1):
        super().__init__()

        self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, 2, 1, groups=in_channels),
                nn.Conv3d(out_channels, out_channels, 1),
                nn.InstanceNorm3d(out_channels),
                nn.GELU()
            )

        self.mamba_blocks = MambaBlock(
            dim=out_channels,
            depth=depth,
            drop_path=drop_path
        )
    
    def forward(self, x):
        x = self.downsample(x)
        x = self.mamba_blocks(x)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels, depth=2):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, 2),
            nn.InstanceNorm3d(out_channels),
            nn.GELU()
        )

        self.skip_conv = nn.Sequential(
            nn.Conv3d(skip_channels + out_channels, out_channels, 1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU()
        )

        self.att = MSA(out_channels)

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
        x = x.view(batch_size, groups, channels_per_group, d, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, d, h, w)
        return x
    
    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.skip_conv(self._channel_shuffle(x, 2))
        x = self.att(x)
        x = self.conv_blocks(x) + x
        return x

class MCVUNet(nn.Module):
    """基于Mamba和卷积并行的3D医学图像分割网络"""
    
    def __init__(self, in_channels=4, out_channels=4, base_dim=32, depths=[2, 2, 2, 2]):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_dim, 7, 1, 3),
            nn.InstanceNorm3d(base_dim),
            nn.GELU()
        )
        

        self.encoders = nn.ModuleList()
        dims = [base_dim, base_dim*2, base_dim*4, base_dim*8, base_dim*16]

        for i in range(4):
            downsample = (i < 4)
            
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

        self.decoders = nn.ModuleList()
        for i in range(3, -1, -1):

            self.decoders.append(
                DecoderBlock(
                    in_channels=dims[i+1],
                    skip_channels=dims[i],
                    out_channels=dims[i],
                    depth=2
                )
            )
        self.out2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(base_dim*2, base_dim, 3, 1, 1,groups=base_dim),
            nn.Conv3d(base_dim, out_channels, 1)
        )
        self.out3 = nn.Sequential(

            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True),
            nn.Conv3d(base_dim*4, base_dim, 3, 1, 1,groups=base_dim),
            nn.Conv3d(base_dim, out_channels, 1)
        )
        self.out4 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True),
            nn.Conv3d(base_dim*8, base_dim, 3, 1, 1,groups=base_dim),
            nn.Conv3d(base_dim, out_channels, 1)
        )
        
        # 最终输出层 - 直接输出，不需要上采样
        self.output_conv = nn.Sequential(
            nn.Conv3d(base_dim, base_dim//2, 3, 1, 1,groups=base_dim//2),
            nn.Conv3d(base_dim//2, base_dim//2, 1),
            nn.InstanceNorm3d(base_dim//2),
            nn.GELU(),
            nn.Conv3d(base_dim//2, out_channels, 1)
            
        )

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
        input_size = x.shape[2:]
        x0 = self.stem(x)
        skips = [x0]
        x = x0
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < 3:
                skips.append(x)

        outs = []

        for i, decoder in enumerate(self.decoders):
            skip = skips[3-i]
            x = decoder(x, skip)
            if i < 3:
                outs.append(x)

        x = self.output_conv(x)
        out2 = self.out2(outs[2])
        out3 = self.out3(outs[1])
        out4 = self.out4(outs[0])
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='trilinear', align_corners=False)
        
        return [x, out2, out3, out4]

if __name__ == "__main__":

    model = MCVUNet(in_channels=1, out_channels=9, base_dim=32)
    model = model.cuda()

    x = torch.randn(1, 1, 96, 96, 96).cuda()

    with torch.no_grad():
        output = model(x)[0]
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


    if x.shape == output.shape:
        print("✓ 输入输出尺寸匹配!")
    else:
        print("✗ 输入输出尺寸不匹配!")

    from thop import profile
    
    def count_flops_and_params(model, input_shape):
        model.eval()
        input_tensor = torch.randn(*input_shape).cuda()
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        return flops, params

    input_shape = (1, 1, 96, 96, 96)
    total_flops, total_params = count_flops_and_params(model, input_shape)
    print(f"模型FLOPs: {total_flops / 1e9:.2f} GFLOPs")
    print(f"thop计算参数量: {total_params / 1e6:.2f}M")

    manual_params = sum(p.numel() for p in model.parameters())
    if abs(total_params - manual_params) < 100:
        print("✓ 参数量计算一致")
    else:
        print(f"⚠ 参数量计算有差异: 手动{manual_params}, thop{total_params}")
    
    
    
    