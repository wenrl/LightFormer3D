import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv3D(nn.Module):

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        if p is None:
            p = k // 2
        self.dw = nn.Conv3d(
            in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=bias
        )
        self.pw = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class DSConvINAct3D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = DSConv3D(in_ch, out_ch, k=k, s=s, p=p, bias=bias)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))



class Stem3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DSConv3D(in_ch, out_ch, k=3, s=1, p=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))



class FeatureBlock3D(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = DSConvINAct3D(in_ch, out_ch, k=3, s=1, p=1)
        self.conv2 = DSConvINAct3D(out_ch, out_ch, k=3, s=1, p=1)
        if in_ch != out_ch:
            self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.conv2(self.conv1(x))
        return out + identity

class Downsample3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = DSConvINAct3D(in_ch, out_ch, k=3, s=2, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class ChannelAttention3D(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        mid = max(1, ch // reduction)
        self.mlp = nn.Sequential(
            nn.Conv3d(ch, mid, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(mid, ch, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool3d(x, output_size=1)
        mx = F.adaptive_max_pool3d(x, output_size=1)
        w = self.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * w


class SpatialAttention3D(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=k, padding=p, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg, mx], dim=1)
        w = self.sigmoid(self.conv(a))
        return x * w


class CBAM3D(nn.Module):
    def __init__(self, ch: int, reduction: int = 8, spatial_k: int = 7):
        super().__init__()
        self.ca = ChannelAttention3D(ch, reduction=reduction)
        self.sa = SpatialAttention3D(k=spatial_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class SimCAM3D(nn.Module):

    def __init__(
        self,
        learnable_scale=False,
        e_lambda=1e-4,
        local_kernel_size=3,
        return_coefficients=False
    ):
        super(SimCAM3D, self).__init__()

        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.local_kernel_size = local_kernel_size
        self.return_coefficients = return_coefficients
        self.branch_logits = nn.Parameter(torch.zeros(2))

    def _simam_energy(self, x):

        b, c, d, h, w = x.size()

        n = d * h * w - 1
        n = max(n, 1)

        x_mean = x.mean(dim=[2, 3, 4], keepdim=True)
        x_minus_mu_square = (x - x_mean).pow(2)

        var = x_minus_mu_square.sum(
            dim=[2, 3, 4],
            keepdim=True
        ) / n

        y = x_minus_mu_square / (4 * (var + self.e_lambda)) + 0.5

        return y

    def _local_structure_compensation(self, x):

        k = self.local_kernel_size
        pad = k // 2

        local_mean = F.avg_pool3d(
            x,
            kernel_size=k,
            stride=1,
            padding=pad,
            count_include_pad=False
        )

        local_x2_mean = F.avg_pool3d(
            x * x,
            kernel_size=k,
            stride=1,
            padding=pad,
            count_include_pad=False
        )

        local_var = local_x2_mean - local_mean.pow(2)
        local_var = torch.clamp(local_var, min=0.0)

        local_dev = torch.abs(x - local_mean)

        structure = local_dev / torch.sqrt(local_var + self.e_lambda)

        structure = structure / (
            structure.mean(dim=[2, 3, 4], keepdim=True) + self.e_lambda
        )

        structure = torch.tanh(structure)

        return structure

    def forward(self, x):

        y_simam = self._simam_energy(x)

        structure = self._local_structure_compensation(x)

        branch_weights = torch.softmax(self.branch_logits, dim=0)

        alpha_global = branch_weights[0]
        alpha_local = branch_weights[1]

        y = alpha_global * y_simam + alpha_local * structure

        attention = self.activation(y)

        # if self.return_coefficients:
        #     return attention, {
        #         "alpha_global": alpha_global,
        #         "alpha_local": alpha_local
        #     }

        return attention



class LiteMHSA3D(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int = 4,
                 kv_stride: int = 2,
                 q_chunk: int = 2048,
                 use_simam: bool = True,
                 simam_learnable_scale: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kv_stride = kv_stride
        self.q_chunk = q_chunk
        self.use_simam = use_simam


        self.q_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=False)

        if kv_stride > 1:
            self.kv_down = nn.Conv3d(
                dim, dim, kernel_size=kv_stride, stride=kv_stride,
                padding=0, groups=dim, bias=False
            )
        else:
            self.kv_down = nn.Identity()

        if use_simam:
            self.simam = SimCAM3D(learnable_scale=simam_learnable_scale)

        self.kv_proj = nn.Conv3d(dim, dim * 2, kernel_size=1, bias=False)

        self.out_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=False)

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        n = d * h * w

        q = self.q_proj(x).flatten(2).transpose(1, 2).contiguous()   # (B, N, C)

        x_kv = x
        if self.use_simam:
            weight = self.simam(x_kv)        # (B, C, D, H, W)
            x_kv = x_kv * weight

        x_kv = self.kv_down(x_kv)            # (B, C, D/k, H/k, W/k)

        kv = self.kv_proj(x_kv)
        k, v = torch.chunk(kv, 2, dim=1)     # 各 (B, C, D/k, H/k, W/k)

        k = k.flatten(2).transpose(1, 2).contiguous()   # (B, M, C)
        v = v.flatten(2).transpose(1, 2).contiguous()   # (B, M, C)

        qh = self._to_heads(q)               # (B, h, N, dh)
        kh = self._to_heads(k)               # (B, h, M, dh)
        vh = self._to_heads(v)               # (B, h, M, dh)


        out = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(b, n, c)   # (B, N, C)
        out = out.transpose(1, 2).contiguous().view(b, c, d, h, w)  # (B, C, D, H, W)

        return self.out_proj(out)


class DepthwiseSeparableMLP3D(nn.Module):

    def __init__(self, dim: int, expansion: float = 2.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.pw1 = nn.Conv3d(dim, hidden, kernel_size=1, bias=False)
        self.pw2 = nn.Conv3d(hidden, dim, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.pw1(x))
        x = self.drop(x)
        x = self.pw2(x)
        x = self.drop(x)
        return x


class LiteTransformerBlock3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        kv_stride: int = 2,
        q_chunk: int = 2048,
        mlp_expansion: float = 2.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(dim, affine=True)
        self.norm2 = nn.InstanceNorm3d(dim, affine=True)

        self.pos = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.attn = LiteMHSA3D(dim, num_heads=num_heads, kv_stride=kv_stride, q_chunk=q_chunk)
        self.mlp = DepthwiseSeparableMLP3D(dim, expansion=mlp_expansion, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x



class UNETRConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = DSConvINAct3D(in_ch, out_ch, k=3, s=1, p=1)
        self.conv2 = DSConvINAct3D(out_ch, out_ch, k=3, s=1, p=1)
        if in_ch != out_ch:
            self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.conv2(self.conv1(x))
        return out + identity


class UNETRUpBlock3D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.block = UNETRConvBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)



class Lite3DSegNet(nn.Module):
    def __init__(self, in_channels: int = 9, out_channels: int = 4, base_dim: int = 32):
        super().__init__()
        b = base_dim


        self.stem = Stem3D(in_channels, b)
        self.stem_feat = FeatureBlock3D(b, b)

        self.down0 = Downsample3D(b, b)
        self.enc0_feat = FeatureBlock3D(b, b)
        self.tr0 = LiteTransformerBlock3D(
            dim=1 * b, num_heads=8, kv_stride=16, q_chunk=2048, mlp_expansion=2.0, drop=0.0
        )

        self.down1 = Downsample3D(b, 2 * b)
        self.enc1_feat = FeatureBlock3D(2 * b, 2 * b)
        self.tr1 = LiteTransformerBlock3D(
            dim=2 * b, num_heads=8, kv_stride=8, q_chunk=1024, mlp_expansion=2.0, drop=0.0
        )

        self.down2 = Downsample3D(2 * b, 4 * b)
        self.enc2_feat = FeatureBlock3D(4 * b, 4 * b)
        self.tr2 = LiteTransformerBlock3D(
            dim=4 * b, num_heads=8, kv_stride=4, q_chunk=512, mlp_expansion=2.0, drop=0.0
        )

        self.down3 = Downsample3D(4 * b, 8 * b)
        self.enc3_feat = FeatureBlock3D(8 * b, 8 * b)
        self.tr3 = LiteTransformerBlock3D(
            dim=8 * b, num_heads=8, kv_stride=2, q_chunk=256, mlp_expansion=2.0, drop=0.0
        )


        self.up3 = UNETRUpBlock3D(in_ch=8 * b, skip_ch=4 * b, out_ch=4 * b)
        self.up2 = UNETRUpBlock3D(in_ch=4 * b, skip_ch=2 * b, out_ch=2 * b)
        self.up1 = UNETRUpBlock3D(in_ch=2 * b, skip_ch=b,     out_ch=b)
        self.up0 = UNETRUpBlock3D(in_ch=b,     skip_ch=b,     out_ch=b)

        self.head1 = nn.Conv3d(b, out_channels, kernel_size=1, bias=True)
        self.head2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(b*1, b, 3, 1, 1,groups=base_dim),
            nn.Conv3d(b, out_channels, 1)
        )
        self.head3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True),
            nn.Conv3d(b*2, b, 3, 1, 1,groups=base_dim),
            nn.Conv3d(b, out_channels, 1)
        )
        self.head4 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True),
            nn.Conv3d(b*4, b, 3, 1, 1,groups=base_dim),
            nn.Conv3d(b, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        s_stem = self.stem_feat(x)

        x0 = self.down0(s_stem)
        s0 = self.enc0_feat(self.tr0(x0))

        x1 = self.down1(s0)
        s1 = self.enc1_feat(self.tr1(x1))

        x2 = self.down2(s1)
        s2 = self.enc2_feat(self.tr2(x2))

        x3 = self.down3(s2)
        s3 = self.enc3_feat(self.tr3(x3)) 



        d2 = self.up3(s3, s2)                               # (B, 4b, 12)
        d1 = self.up2(d2, s1)                               # (B, 2b, 24)
        d0 = self.up1(d1, s0)                               # (B,  b, 48)
        d  = self.up0(d0, s_stem)                           # (B,  b, 96)

        out1 = self.head1(d)                                  # (B, out_channels, 96)
        out2 = self.head2(d0)                                  # (B, out_channels, 96)
        out3 = self.head3(d1)                                  # (B, out_channels, 96)
        out4 = self.head4(d2)                                  # (B, out_channels, 96)
        return [out1, out2, out3, out4] #training



def count_params_manual(model: nn.Module, verbose: bool = True) -> int:
    total = 0
    if verbose:
        print("========== Parameter breakdown ==========")
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if verbose:
            print(f"{name:70s} {n:12d}")
    if verbose:
        print("========================================")
        print(f"Total params (manual) = {total}")
    return total


def _get_dev(y):
    if isinstance(y, (tuple, list)):
        y = y[0]
    return y.device


def zero_ops(m, x, y):
    dev = _get_dev(y)
    if not hasattr(m, "total_ops") or m.total_ops.device != dev:
        m.total_ops = torch.zeros(1, device=dev)
    m.total_ops += torch.zeros(1, device=dev)


def lite_mhsa_ops(m: LiteMHSA3D, x, y):
    inp = x[0]
    b, c, d, h, w = inp.shape
    n = d * h * w

    if m.kv_stride > 1:
        dk = d // m.kv_stride
        hk = h // m.kv_stride
        wk = w // m.kv_stride
    else:
        dk, hk, wk = d, h, w
    mm = dk * hk * wk

    flops = 4 * b * n * mm * c

    dev = _get_dev(y)
    if not hasattr(m, "total_ops") or m.total_ops.device != dev:
        m.total_ops = torch.zeros(1, device=dev)
    m.total_ops += torch.tensor([flops], device=dev, dtype=torch.float32)


def safe_profile_thop(model: nn.Module, input_tensor: torch.Tensor):
    from thop import profile

    model_p = copy.deepcopy(model).cpu().eval()
    x_cpu = input_tensor.detach().cpu()

    custom_ops = {
        nn.InstanceNorm3d: zero_ops,
        nn.GELU: zero_ops,
        LiteMHSA3D: lite_mhsa_ops,
    }

    with torch.no_grad():
        flops, params_thop = profile(model_p, inputs=(x_cpu,), custom_ops=custom_ops, verbose=False)
    return flops, params_thop

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    model = Lite3DSegNet(in_channels=9, out_channels=4, base_dim=32)
    x = torch.randn(1, 9, 96, 96, 96)
    y = model(x)[0]
    print("Output shape:", tuple(y.shape))  # (1, 4, 96, 96, 96)

    total_params = count_params_manual(model, verbose=False)
    flops, _ = safe_profile_thop(model, x)

    print(f"模型FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"手动参数量: {total_params / 1e6:.2f} M")
