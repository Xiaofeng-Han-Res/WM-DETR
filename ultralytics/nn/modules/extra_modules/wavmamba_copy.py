import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== Haar DWT / iDWT（四子带） ====================

def _haar_kernels(device, dtype):
    ll = torch.tensor([[0.5,  0.5],
                       [0.5,  0.5]], device=device, dtype=dtype)
    lh = torch.tensor([[0.5, -0.5],
                       [0.5, -0.5]], device=device, dtype=dtype)
    hl = torch.tensor([[0.5,  0.5],
                       [-0.5, -0.5]], device=device, dtype=dtype)
    hh = torch.tensor([[0.5, -0.5],
                       [-0.5,  0.5]], device=device, dtype=dtype)
    return torch.stack([ll, lh, hl, hh], dim=0)  # (4,2,2)

class DWT2DSplit(nn.Module):
    """(B,C,H,W) -> (LL,LH,HL,HH)，各 (B,C,H/2,W/2)。自动补齐偶数边。"""
    def forward(self, x):
        B, C, H, W = x.shape
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
        x32 = x.float()
        K = _haar_kernels(x32.device, x32.dtype).view(4, 1, 2, 2)  # (4,1,2,2)
        w = K.repeat(C, 1, 1, 1)                                   # (4C,1,2,2)
        y = F.conv2d(x32, w, stride=2, padding=0, groups=C)         # (B,4C,H/2,W/2)
        y = y.to(x.dtype)
        H2, W2 = H // 2, W // 2
        y = y.view(B, 4, C, H2, W2)
        return (y[:, 0], y[:, 1], y[:, 2], y[:, 3]), (pad_h, pad_w)

class IDWT2DSplit(nn.Module):
    """(LL,LH,HL,HH) -> (B,C,H,W)，自动裁掉上一步 pad。"""
    def forward(self, bands, pads):
        LL, LH, HL, HH = bands
        B, C, H2, W2 = LL.shape
        y  = torch.stack([LL, LH, HL, HH], dim=1).view(B, 4*C, H2, W2)  # (B,4C,H2,W2)
        y32 = y.float()

        # 直接 repeat 得到 (4C,1,2,2)，groups=C
        K = _haar_kernels(y32.device, y32.dtype).view(4, 1, 2, 2)       # (4,1,2,2)
        w = K.repeat(C, 1, 1, 1)                                        # (4C,1,2,2)

        # conv_transpose2d 规则：weight=(in_ch, out_ch/groups, kH, kW)
        # 这里 in_ch=4C, out_ch=C, groups=C => out_ch/groups = 1 ✔
        x = F.conv_transpose2d(y32, w, stride=2, padding=0, groups=C).to(LL.dtype)

        pad_h, pad_w = pads
        if pad_h or pad_w:
            x = x[:, :, : x.shape[2]-pad_h, : x.shape[3]-pad_w].contiguous()
        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

# ==================== FP32 LayerNorm（AMP 友好） ====================

class FP32LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # 在 FP32 中做 LN，避免 AMP 半精度下的 dtype 冲突
        w = self.weight.float() if self.weight is not None else None
        b = self.bias.float() if self.bias is not None else None
        y = F.layer_norm(x.float(), x.shape[-1:], w, b, self.eps)
        return y.to(x.dtype)

# ==================== Mamba 核（mamba-ssm 可选；否则轻量回退） ====================

class MambaOrLite(nn.Module):
    """
    (B,L,C)->(B,L,C)
    - 若已安装 mamba-ssm：使用官方 Mamba；
    - 否则使用轻量回退（GLU + 深度可分离 Conv1d），保持长度不变（SAME padding）。
    """
    def __init__(self, embed, d_state=16, d_conv=3, expand=2, dropout=0.0):
        super().__init__()
        try:
            import mamba_ssm  # noqa
            self.use_mamba = True
            # print('true')
        except Exception:
            self.use_mamba = False
            # print('false')

        self.drop = nn.Dropout(dropout)
        if self.use_mamba:
            from mamba_ssm import Mamba
            self.core = Mamba(d_model=embed, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.ln = FP32LayerNorm(embed, eps=1e-6)
            hid = embed * expand
            self.in_proj = nn.Linear(embed, hid * 2, bias=True)   # GLU
            self.dw = nn.Conv1d(hid, hid, kernel_size=d_conv, padding=0, groups=hid, bias=True)
            self.out = nn.Linear(hid, embed, bias=True)

    def forward(self, x):
        x_dtype = x.dtype
        if self.use_mamba:
            core_dtype = next(self.core.parameters()).dtype
            y = self.core(x.to(core_dtype))
            y = y.to(x_dtype)
            return self.drop(torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4))

        # 轻量回退：LN -> GLU -> DWConv1d(SAME) -> Linear -> Residual
        x0 = x
        x  = self.ln(x)
        x  = self.in_proj(x)
        a, b = x.chunk(2, dim=-1)
        x  = a * torch.sigmoid(b)              # GLU（有界）
        x  = x.transpose(1, 2)                 # (B,Hid,L)
        k  = self.dw.kernel_size[0]
        pad_l, pad_r = (k - 1) // 2, k // 2    # SAME 长度（奇偶核均不改 L）
        x  = F.pad(x, (pad_l, pad_r))
        x  = self.dw(x).transpose(1, 2)        # (B,L,Hid)
        x  = self.out(x)
        x  = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        return self.drop(x + x0)

# ==================== 2D 封装：Global / Window Mamba ====================

class GlobalMamba2d(nn.Module):
    """把 (B,C,H,W) 展平成 (B,L,C) 做全局 Mamba，再还原。"""
    def __init__(self, C, d_state=16, d_conv=3, expand=2, dropout=0.0, use_ln=True):
        super().__init__()
        self.ln = FP32LayerNorm(C, eps=1e-6) if use_ln else nn.Identity()
        self.mamba = MambaOrLite(C, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)

    def forward(self, x):  # (B,C,H,W)
        B, C, H, W = x.shape
        seq = self.ln(x.flatten(2).transpose(1, 2).contiguous())  # (B,L,C)
        seq = self.mamba(seq)
        y   = seq.transpose(1, 2).contiguous().view(B, C, H, W)
        return torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)

def _win_part(x, win):
    # x: (B,C,H,W) -> (B*nW, C, win, win)，并返回 Hp, Wp, pad
    B, C, H, W = x.shape
    pad_h = (win - H % win) % win
    pad_w = (win - W % win) % win
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
        H, W = H + pad_h, W + pad_w
    x = x.view(B, C, H // win, win, W // win, win)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, C, win, win)
    return x, H, W, pad_h, pad_w

def _win_merge(xw, win, H, W, pad_h, pad_w, B):
    # xw: (B*nW, C, win, win) -> (B,C,H,W)
    C = xw.shape[1]
    xw = xw.view(B, H // win, W // win, C, win, win).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
    if pad_h or pad_w:
        xw = xw[:, :, : H - pad_h, : W - pad_w].contiguous()
    return xw

class WindowMamba2d(nn.Module):
    """窗口 Mamba：每个 win×win 窗口展成序列做 Mamba。"""
    def __init__(self, C, win=7, d_state=16, d_conv=3, expand=2, dropout=0.0, use_ln=True):
        super().__init__()
        self.win = int(win)
        self.ln = FP32LayerNorm(C, eps=1e-6) if use_ln else nn.Identity()
        self.mamba = MambaOrLite(C, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        xw, Hp, Wp, ph, pw = _win_part(x, self.win)                # (B*nW,C,win,win)
        Bn = xw.shape[0]
        seq = self.ln(xw.flatten(2).transpose(1, 2).contiguous())  # (B*nW,L,C), L=win*win
        seq = self.mamba(seq)
        xw  = seq.transpose(1, 2).contiguous().view(Bn, C, self.win, self.win)
        y   = _win_merge(xw, self.win, Hp, Wp, ph, pw, B)
        return torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)

# ==================== 单尺度门控 + 主模块 ====================

# class IntraScaleGate(nn.Module):
#     """仅用 LL 生成四带权重 (B,4,H/2,W/2)。"""
#     def __init__(self, C):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(C, C, 1, bias=True),
#             nn.GELU(),
#             nn.Conv2d(C, 4, 1, bias=True)
#         )

#     def forward(self, LL):                 # (B,C,H2,W2)
#         logits = self.net(LL)              # (B,4,H2,W2)
#         # 为稳定性，softmax 前做 clamp
#         return torch.softmax(logits.clamp(-20, 20), dim=1)

# class IntraScaleGate(nn.Module):
#     """仅用 LL 生成四带权重 (B,4,H/2,W/2)，基线=1，输出范围约 (0, 2)。"""
#     def __init__(self, C):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(C, C, 1, bias=True),
#             nn.GELU(),
#             nn.Conv2d(C, 4, 1, bias=True)
#         )
#         # 零初始化：初始权重≈1（即不改变任何子带的幅度）
#         nn.init.zeros_(self.net[-1].weight)
#         nn.init.zeros_(self.net[-1].bias)

#     def forward(self, LL):                 # (B,C,H2,W2)
#         delta = torch.sigmoid(self.net(LL)) * 2.0   # (0,2)
#         return 1.0 + (delta - 1.0)                  # 中心在1，范围(0,2)
    
class IntraScaleGate(nn.Module):
    """
    改进版：引入水下清晰度先验 (contrast/visibility)，
    在 LL 基础上自适应生成四带权重 (B,4,H/2,W/2)。
    """
    def __init__(self, C):
        super().__init__()
        # 通道扩展：LL(C) + clarity(1) -> C+1
        self.net = nn.Sequential(
            nn.Conv2d(C + 1, C, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(C, 4, 1, bias=True)
        )
        # 零初始化：初始权重≈1
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, LL):   # (B,C,H2,W2)
        # ---- 计算清晰度先验 ----
        # 方差/标准差作为对比度指标，代表水下可见度
        clarity = LL.var(dim=1, keepdim=True).sqrt()   # (B,1,H2,W2)

        # ---- 拼接后送入门控 ----
        x = torch.cat([LL, clarity], dim=1)            # (B,C+1,H2,W2)
        delta = torch.sigmoid(self.net(x)) * 2.0       # (0,2)
        return 1.0 + (delta - 1.0)                     # 中心=1，范围(0,2)


class WavMamba2d(nn.Module):
    """
    单尺度：输入/输出 (B,C,H,W) 不变。
      1) DWT → (LL,LH,HL,HH)
      2) LL → GlobalMamba2d；细节带 → WindowMamba2d
      3) 用 LL 产生门控权重加权四带
      4) iDWT 重建 + 1x1 投影 + 残差
    """
    def __init__(self, C,
                 win_detail=7, d_state=16, d_conv=3, expand=2,
                 dropout=0.0, use_ln=True):
        super().__init__()
        self.dwt  = DWT2DSplit()
        self.idwt = IDWT2DSplit()
        self.g_mamba = GlobalMamba2d(C, d_state=d_state, d_conv=d_conv,
                                     expand=expand, dropout=dropout, use_ln=use_ln)
        self.w_mamba = WindowMamba2d(C, win=win_detail, d_state=d_state, d_conv=d_conv,
                                     expand=expand, dropout=dropout, use_ln=use_ln)
        self.gate = IntraScaleGate(C)
        self.out_proj = nn.Conv2d(C, C, 1, bias=True)

    def forward(self, x):                          # x: (B,C,H,W)
        # 1) DWT
        (LL, LH, HL, HH), pads = self.dwt(x)       # 各 (B,C,H/2,W/2)

        # 2) 频带处理
        LLp = self.g_mamba(LL)                     # 全局稳定
        LHp = self.w_mamba(LH)                     # 细节增强
        HLp = self.w_mamba(HL)
        HHp = self.w_mamba(HH)

        # 3) 门控（来自 LLp）
        gate = self.gate(LLp)                      # (B,4,H/2,W/2)
        gLL, gLH, gHL, gHH = gate[:,0:1], gate[:,1:2], gate[:,2:3], gate[:,3:4]

        # 4) 融合 + iDWT + 残差
        LLf, LHf, HLf, HHf = LLp*gLL, LHp*gLH, HLp*gHL, HHp*gHH
        x_rec = self.idwt((LLf, LHf, HLf, HHf), pads)   # (B,C,H,W)
        y = x + self.out_proj(x_rec)
        return torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)

__all__ = [
    "WavMamba2d",
    "GlobalMamba2d", "WindowMamba2d",
    "DWT2DSplit", "IDWT2DSplit", "FP32LayerNorm", "MambaOrLite"
]
