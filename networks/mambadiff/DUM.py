import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# import torch.nn as nn
# import torch 
# from functools import partial
# from einops import rearrange
# from monai.networks.blocks.dynunet_block import UnetOutBlock
# from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
# from mamba_ssm import Mamba
# import torch.nn.functional as F 
from typing import Optional, Sequence, Union
import math 
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep
from monai.inferers import SlidingWindowInferer
from mamba_ssm import Mamba
from diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.resample import UniformSampler


def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.001] = 0.001
    uncer_out = - pred_out * torch.log(pred_out)
    return uncer_out
def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    return x*torch.sigmoid(x)

class SimAM(torch.nn.Module):
    def __init__(self,channels=None,e_lambda=1e-4):
        super(SimAM,self).__init__()
        self.activaton =nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)

    def forward(self,x):
        b,c,l,h,w = x.size()
        n = l * h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3,4],keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3,4],keepdim=True) / n + self.e_lambda)) + 0.5
        return x*self.activaton(y)


class ImageConv(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            dropout: Union[float, tuple] = 0.0,
            dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)

class LabelConv(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()
        self.temb_proj = torch.nn.Linear(512,
                                         out_chns)
        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims,in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
             spatial_dims,out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)
    
    def forward(self, x, temb):
        x = self.conv_0(x)
        x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        x = self.conv_1(x)
        return x 

class DecodeConv(nn.Module):

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = LabelConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor], temb):
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1), temb)  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0, temb)

        return x

class SSMlayerLabel(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        self.simam = SimAM()
        self.temb_proj = torch.nn.Linear(512,
                                         input_dim)

    def forward(self, x, temb):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        x_simam = self.simam(x)
        x_simam = x_simam.transpose(1, 4)
        x_simam = self.norm(x_simam)
        x_simam = x_simam.transpose(1, 4)
        x = self.max_pool(x_simam)
        x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        B, C, D, H, W = x.shape
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        n_tokens1 = x.shape[3:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B , C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + x_flat * self.skip_scale
        x_mamba1 = self.norm(x_mamba)
        x_mamba1 = x_mamba1.transpose(-1, -2).reshape(B, C , *img_dims)
        x_flat1 = x_mamba1.reshape(B * D, C, n_tokens1).transpose(-1, -2)
        x_mamba1 = self.mamba(x_flat1)
        x_mamba1 = x_mamba1.reshape(B,n_tokens,C) + x_mamba * self.skip_scale
        x_mamba = self.proj(x_mamba1)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out

class SMMLayerImage(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        self.simam = SimAM()
        self.temb_proj = torch.nn.Linear(512,
                                         input_dim)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        x_simam = self.simam(x)
        x_simam = x_simam.transpose(1,4)
        x_simam = self.norm(x_simam)
        x_simam = x_simam.transpose(1, 4)
        x = self.max_pool(x_simam)

        B, C, D, H, W = x.shape
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        n_tokens1 = x.shape[3:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + x_flat * self.skip_scale

        x_mamba1 = self.norm(x_mamba)
        x_mamba1 = x_mamba1.transpose(-1, -2).reshape(B, C, *img_dims)
        x_flat1 = x_mamba1.reshape(B * D, C, n_tokens1).transpose(-1, -2)
        x_mamba1 = self.mamba(x_flat1)
        x_mamba1 = x_mamba1.reshape(B, n_tokens, C) + x_mamba * self.skip_scale

        x_mamba = self.proj(x_mamba1)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


class ImageEncoder(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):

        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.conv0 = ImageConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down1 = SMMLayerImage(fea[0], fea[1])
        self.down2 = SMMLayerImage(fea[1], fea[2])
        self.down3 = SMMLayerImage(fea[2], fea[3])
        self.down4 = SMMLayerImage(fea[3], fea[4])

    def forward(self, x: torch.Tensor):
        x0 = self.conv0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        return [x0, x1, x2, x3, x4]

class LabelUnet(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):

        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")
        
        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])

        self.conv0 = LabelConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down1 = SSMlayerLabel(fea[0], fea[1])
        self.down2 = SSMlayerLabel(fea[1], fea[2])
        self.down3 = SSMlayerLabel(fea[2], fea[3])
        self.down4 = SSMlayerLabel(fea[3], fea[4])

        self.up4 = DecodeConv(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.up3 = DecodeConv(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.up2 = DecodeConv(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.up1 = DecodeConv(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)



    def forward(self, x: torch.Tensor, t, embeddings=None, image=None):

        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        x0 = self.conv0(x, temb)
        if embeddings is not None:
            x0 += embeddings[0]

        x1 = self.down1(x0, temb)
        if embeddings is not None:
            x1 += embeddings[1]

        x2 = self.down2(x1, temb)
        if embeddings is not None:
            x2 += embeddings[2]

        x3 = self.down3(x2, temb)
        if embeddings is not None:
            x3 += embeddings[3]

        x4 = self.down4(x3, temb)
        if embeddings is not None:
            x4 += embeddings[4]

        u4 = self.up4(x4, x3, temb)
        u3 = self.up3(u4, x2, temb)
        u2 = self.up2(u3, x1, temb)
        u1 = self.up1(u2, x0, temb)

        logits = self.final_conv(u1)

        return logits
        # return [logits,x3]


class DUM(nn.Module):
    def __init__(self,number_modality,number_targets) -> None:
        super().__init__()
        self.embed_model = ImageEncoder(3, number_modality, number_targets, (32, 32, 64, 128, 256, 32))

        self.model = LabelUnet(3, number_targets, number_targets, (32, 32, 64, 128, 256, 32),
                                 act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.MSE,
                                         )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(1000)
        self.number_targets = number_targets

    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, self.number_targets, 96,96,96),
                                                                model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]

            return sample_out


if __name__ == "__main__":
    # 检查GPU是否可用
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"使用设备: {device}")
    
    # 创建模型并移动到GPU
    model = DUM(number_modality=1, number_targets=9)
    model = model.cuda()
    
    # 测试输入
    x = torch.randn(1, 1, 96, 96, 96).cuda()
    y = torch.randn(1, 9, 96, 96, 96)
    y = (y*2-1).cuda()
    SWI = SlidingWindowInferer(roi_size=[96,96,96], sw_batch_size=1, overlap=0.5)
    # 前向传播
    with torch.no_grad():
        output = model(x = y, pred_type = "q_sample")
        output = model(x = output[0], step = output[1], image = x, pred_type = "denoise")
        
        # output = SWI(x, model, pred_type = "ddim_sample") 
    
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
