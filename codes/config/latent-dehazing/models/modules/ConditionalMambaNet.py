import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .module_util import SinusoidalPosEmb, LayerNorm, exists
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import math
import warnings

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ConditionalMambaNet(nn.Module):
    """
    基于MambaIRv2的条件扩散去噪网络
    用于替换Refusion中的ConditionalNAFNet
    """

    def __init__(self,
                 img_channel=8,  # Refusion在潜空间工作，通道数为8
                 width=64,
                 middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 28],
                 dec_blk_nums=[1, 1, 1, 1],
                 d_state=16,
                 num_heads=4,
                 window_size=8,
                 inner_rank=32,
                 num_tokens=64,
                 upscale=1):
        super().__init__()

        self.upscale = upscale
        self.window_size = window_size

        # 时间嵌入网络
        fourier_dim = width
        time_dim = width * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(fourier_dim),
            nn.Linear(fourier_dim, time_dim * 2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        # 输入投影：将输入和条件拼接后投影
        # img_channel*2 因为我们会拼接 (inp-cond) 和 cond
        self.intro = nn.Conv2d(img_channel * 2, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1)

        # 编码器
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for i, num in enumerate(enc_blk_nums):
            # 计算当前分辨率
            scale_factor = 2 ** i
            curr_resolution = (64 // scale_factor, 64 // scale_factor)  # 假设输入64x64

            layers = []
            for j in range(num):
                layers.append(
                    TimeConditionedAttentiveLayer(
                        dim=chan,
                        d_state=d_state,
                        input_resolution=curr_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if (j % 2 == 0) else window_size // 2,
                        inner_rank=inner_rank,
                        num_tokens=num_tokens,
                        time_emb_dim=time_dim
                    )
                )
            self.encoders.append(nn.ModuleList(layers))

            # 下采样
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan = chan * 2

        # 中间块
        middle_resolution = (64 // (2 ** len(enc_blk_nums)), 64 // (2 ** len(enc_blk_nums)))
        middle_layers = []
        for _ in range(middle_blk_num):
            middle_layers.append(
                TimeConditionedAttentiveLayer(
                    dim=chan,
                    d_state=d_state,
                    input_resolution=middle_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0,
                    inner_rank=inner_rank,
                    num_tokens=num_tokens,
                    time_emb_dim=time_dim
                )
            )
        self.middle_blks = nn.ModuleList(middle_layers)

        # 解码器
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i, num in enumerate(dec_blk_nums):
            # 上采样
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2

            # 计算当前分辨率
            scale_factor = 2 ** (len(dec_blk_nums) - i - 1)
            curr_resolution = (64 // scale_factor, 64 // scale_factor)

            layers = []
            for j in range(num):
                layers.append(
                    TimeConditionedAttentiveLayer(
                        dim=chan,
                        d_state=d_state,
                        input_resolution=curr_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if (j % 2 == 0) else window_size // 2,
                        inner_rank=inner_rank,
                        num_tokens=num_tokens,
                        time_emb_dim=time_dim
                    )
                )
            self.decoders.append(nn.ModuleList(layers))

        self.padder_size = 2 ** len(self.encoders)

        # 初始化相对位置索引
        self.register_buffer('relative_position_index_SA', self._calculate_rpi_sa())

    def _calculate_rpi_sa(self):
        """计算相对位置索引"""
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def _calculate_mask(self, x_size):
        """计算shifted window的attention mask"""
        h, w = x_size
        # 确保尺寸是 window_size 的倍数
        assert h % self.window_size == 0 and w % self.window_size == 0, \
            f"Image size ({h}, {w}) must be divisible by window_size ({self.window_size})"

        img_mask = torch.zeros((1, h, w, 1), device=next(self.parameters()).device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def check_image_size(self, x):
        """确保图像尺寸是padder_size的倍数"""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def forward(self, inp, cond, time):
        """
        Args:
            inp: [B, C, H, W] - 噪声状态
            cond: [B, C, H, W] - 条件（低质量图像的潜编码）
            time: [B] or scalar - 时间步
        Returns:
            [B, C, H, W] - 预测的噪声或去噪后的图像
        """
        # 处理时间输入
        if isinstance(time, (int, float)):
            time = torch.tensor([time], device=inp.device).expand(inp.shape[0])

        # 时间嵌入
        t = self.time_mlp(time)  # [B, time_dim]

        # 保存原始尺寸
        B, C, H, W = inp.shape

        # 计算差异并拼接
        x = inp - cond
        x = torch.cat([x, cond], dim=1)  # [B, 2C, H, W]

        # Padding
        x = self.check_image_size(x)
        _, _, H_pad, W_pad = x.shape
        x_size = (H_pad, W_pad)

        # 计算attention mask和params
        attn_mask = self._calculate_mask((H_pad, W_pad))
        params = {
            'attn_mask': attn_mask,
            'rpi_sa': self.relative_position_index_SA
        }

        # 输入投影
        x = self.intro(x)  # [B, width, H, W]

        # 转换为序列形式用于Transformer
        # [B, C, H, W] -> [B, H*W, C]
        x = rearrange(x, 'b c h w -> b (h w) c')


        # 编码器
        encs = [x]
        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            # 通过AttentiveLayer块
            for layer in encoder:
                x = layer(x, x_size, time_emb=t, params=params)

            encs.append(x)

            # 下采样前转回空间形式
            x = rearrange(x, 'b (h w) c -> b c h w', h=x_size[0], w=x_size[1])
            x = down(x)

            # 更新尺寸
            x_size = (x_size[0] // 2, x_size[1] // 2)
            x = rearrange(x, 'b c h w -> b (h w) c')

            # 更新attention mask
            if x_size[0] >= self.window_size:
                attn_mask = self._calculate_mask(x_size)
                params['attn_mask'] = attn_mask

        # 中间块
        for layer in self.middle_blks:
            x = layer(x, x_size, time_emb=t, params=params)

        for i, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, reversed(encs))):
            # 上采样前转回空间形式
            x = rearrange(x, 'b (h w) c -> b c h w', h=x_size[0], w=x_size[1])
            x = up(x)

            # 更新尺寸
            x_size = (x_size[0] * 2, x_size[1] * 2)

            # 转回序列形式并添加skip connection
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = x + enc_skip

            # 通过AttentiveLayer块
            for layer in decoder:
                x = layer(x, x_size, time_emb=t, params=params)

        # 转回空间形式
        x = rearrange(x, 'b (h w) c -> b c h w', h=x_size[0], w=x_size[1])

        # 输出投影 + 残差连接
        x = self.ending(x + encs[0].reshape(B, H_pad, W_pad, -1).permute(0, 3, 1, 2))

        # 裁剪回原始尺寸
        x = x[..., :H, :W]

        return x

# 时间嵌入的替换模块 ############################################################################################
# 时间嵌入的替换模块 ############################################################################################
# 时间嵌入的替换模块 ############################################################################################
class TimeConditionedAttentiveLayer(nn.Module):
    """
    时间条件化的AttentiveLayer，用于替换NAFBlock
    将时间嵌入集成到AttentiveLayer中，使其适用于扩散模型
    """

    def __init__(self,
                 dim,
                 d_state=16,
                 input_resolution=(32, 32),
                 num_heads=4,
                 window_size=8,
                 shift_size=0,
                 inner_rank=32,
                 num_tokens=64,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 time_emb_dim=None,  # 新增：时间嵌入维度
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.time_emb_dim = time_emb_dim or dim * 4

        # 时间嵌入映射层 - 用于生成自适应调制参数
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, dim * 8)  # 为两个norm和两个FFN生成scale和shift
        ) if time_emb_dim else None

        # Layer normalization
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        # Layer scale参数
        self.scale1 = nn.Parameter(1e-4 * torch.ones(dim), requires_grad=True)
        self.scale2 = nn.Parameter(1e-4 * torch.ones(dim), requires_grad=True)

        # Window attention部分
        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.win_mhsa = WindowAttention(
            self.dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        # ASSM (Attentive State Space) 部分
        self.assm = ASSM(
            self.dim,
            d_state,
            input_resolution=input_resolution,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio
        )

        # ConvFFN部分
        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.convffn1 = GatedMLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim)
        self.convffn2 = GatedMLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim)

        # Token embeddings for ASSM
        self.embeddingA = nn.Embedding(inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / inner_rank, 1 / inner_rank)

    def time_condition(self, time_emb):
        """生成时间条件化的调制参数"""
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)  # [B, dim*8]
            time_emb = rearrange(time_emb, 'b c -> b 1 c')  # [B, 1, dim*8]
            # 分割为6组：每个norm层的scale和shift
            chunks = time_emb.chunk(8, dim=-1)
            return {
                'scale_att': chunks[0],
                'shift_att': chunks[1],
                'scale_ffn1': chunks[2],
                'shift_ffn1': chunks[3],
                'scale_assm': chunks[4],
                'shift_assm': chunks[5],
                'scale_ffn2': chunks[6],
                'shift_ffn2': chunks[7],
            }
        return None

    def forward(self, x, x_size, time_emb=None, params=None):
        """
        Args:
            x: [B, H*W, C]
            x_size: (H, W)
            time_emb: [B, time_emb_dim] 时间嵌入
            params: 包含attn_mask和rpi_sa的字典
        """
        h, w = x_size
        b, n, c = x.shape

        # 获取时间条件
        time_cond = self.time_condition(time_emb)

        # Part 1: Window-based Multi-Head Self-Attention with time conditioning
        shortcut = x
        x = self.norm1(x)

        # 应用时间条件的自适应调制
        if time_cond is not None:
            x = x * (1 + time_cond['scale_att']) + time_cond['shift_att']

        qkv = self.wqkv(x)
        qkv = qkv.reshape(b, h, w, 3 * c)

        # Shifted window attention
        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = params['attn_mask'] if params else None
        else:
            shifted_qkv = qkv
            attn_mask = None

        # Window partition
        x_windows = window_partition(shifted_qkv, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, 3 * c)

        # Attention
        attn_windows = self.win_mhsa(x_windows, rpi=params['rpi_sa'] if params else None, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)

        # Reverse window partition
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x

        x_win = attn_x.view(b, n, c) + shortcut

        # FFN with time conditioning
        norm_x = self.norm2(x_win)
        if time_cond is not None:
            norm_x = norm_x * (1 + time_cond['scale_ffn1']) + time_cond['shift_ffn1']

        x_win = self.convffn1(norm_x, x_size) + x_win
        x = shortcut * self.scale1 + x_win

        # Part 2: Attentive State Space with time conditioning
        shortcut = x
        norm_x = self.norm3(x)

        if time_cond is not None:
            norm_x = norm_x * (1 + time_cond['scale_assm']) + time_cond['shift_assm']

        x_aca = self.assm(norm_x, x_size, self.embeddingA) + x
        norm_x = self.norm4(x_aca)
        if time_cond is not None:
            norm_x = norm_x * (1 + time_cond['scale_ffn2']) + time_cond['shift_ffn2']
        x = x_aca + self.convffn2(norm_x, x_size)
        x = shortcut * self.scale2 + x

        return x

# 保留原有的辅助类 ############################################################################################
# 保留原有的辅助类 ############################################################################################
# 保留原有的辅助类 ############################################################################################
def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x

class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)  # DW Conv

    def forward(self, x, H, W):
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)).flatten(2).transpose(-1,
                                                                                                              -2).contiguous()

        return x1 * x2


class GatedMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r"""
    Shifted Window-based Multi-head Self-Attention

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None):
        r"""
        Args:
            qkv: Input query, key, and value tokens with shape of (num_windows*b, n, c*3)
            rpi: Relative position index
            mask (0/-inf):  Mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'


class ASSM(nn.Module):
    def __init__(self, dim, d_state, input_resolution, num_tokens=64, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)  # [64,32] [32, 48] = [64,48]
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, x_size, token):
        B, n, C = x.shape
        H, W = x_size

        full_embedding = self.embeddingB.weight @ token.weight  # [128, C]

        pred_route = self.route(x)  # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]

        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # b,n,c

        semantic_x = semantic_neighbor(x, x_sort_indices)
        y = self.selectiveScan(semantic_x, prompt)
        y = self.out_proj(self.out_norm(y))
        x = semantic_neighbor(y, x_sort_indices_reverse)

        return x


class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt):
        B, L, C = x.shape
        K = 1  # mambairV2 needs noly 1 scan
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()  # B, 1, C ,L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) + prompt  # (b, k, d_state, l)  our ASE here!
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt, **kwargs):
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt)  # [B, L, C]
        y = y.permute(0, 2, 1).contiguous()
        return y