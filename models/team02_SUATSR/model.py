import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 基础模块
# =====================================================================
class LayerNorm2d(nn.Module):
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.eps = eps

    def forward(self, x):
        var  = x.var(dim=1, keepdim=True, unbiased=False)
        mean = x.mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.w + self.b


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# =====================================================================
# 小波变换
# =====================================================================
class HaarDWT(nn.Module):
    def forward(self, x):
        A = x[:, :, 0::2, 0::2]; B = x[:, :, 0::2, 1::2]
        C = x[:, :, 1::2, 0::2]; D = x[:, :, 1::2, 1::2]
        return torch.cat([(A+B+C+D)*.5, (A-B+C-D)*.5,
                          (A+B-C-D)*.5, (A-B-C+D)*.5], dim=1)


class HaarIDWT(nn.Module):
    def forward(self, x):
        nc = x.shape[1] // 4
        LL, LH, HL, HH = x[:, :nc], x[:, nc:2*nc], x[:, 2*nc:3*nc], x[:, 3*nc:]
        B2, _, H2, W2 = x.shape
        out = torch.empty(B2, nc, H2*2, W2*2, device=x.device, dtype=x.dtype)
        out[:, :, 0::2, 0::2] = (LL+LH+HL+HH) * .5
        out[:, :, 0::2, 1::2] = (LL-LH+HL-HH) * .5
        out[:, :, 1::2, 0::2] = (LL+LH-HL-HH) * .5
        out[:, :, 1::2, 1::2] = (LL-LH-HL+HH) * .5
        return out


# =====================================================================
# HAT-style 注意力模块
# =====================================================================
class WindowMSA(nn.Module):
    def __init__(self, ch, ws, nh):
        super().__init__()
        self.ws = ws; self.nh = nh; self.hd = ch // nh
        self.scale = self.hd ** -0.5
        self.qkv  = nn.Linear(ch, ch*3, bias=False)
        self.proj = nn.Linear(ch, ch,   bias=False)
        self.rpe  = nn.Parameter(torch.zeros((2*ws-1)**2, nh))
        nn.init.trunc_normal_(self.rpe, std=0.02)
        self.register_buffer('rpe_idx', self._idx(ws))

    @staticmethod
    def _idx(ws):
        gy, gx = torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing='ij')
        flat = torch.stack([gy.flatten(), gx.flatten()])
        rel  = flat[:, :, None] - flat[:, None, :]
        rel[0] += ws-1; rel[1] += ws-1; rel[0] *= 2*ws-1
        return rel.sum(0)

    def forward(self, x):
        Bnw, N, C = x.shape
        qkv = self.qkv(x).reshape(Bnw, N, 3, self.nh, self.hd)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rpe[self.rpe_idx].permute(2, 0, 1).unsqueeze(0)
        return self.proj((attn.softmax(-1) @ v).transpose(1, 2).reshape(Bnw, N, C))


class OverlappingCrossAttn(nn.Module):
    def __init__(self, ch, ws, ov, nh):
        super().__init__()
        self.ws = ws; self.ov = ov
        self.nh = nh; self.hd = ch // nh
        self.scale = self.hd ** -0.5
        self.q    = nn.Linear(ch, ch,   bias=False)
        self.kv   = nn.Linear(ch, ch*2, bias=False)
        self.proj = nn.Linear(ch, ch,   bias=False)
        self.norm = nn.LayerNorm(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws, ov = self.ws, self.ov
        ph = (ws - H % ws) % ws; pw = (ws - W % ws) % ws
        xp = F.pad(x, (0, pw, 0, ph), mode='reflect') if (ph or pw) else x
        _, _, Hp, Wp = xp.shape
        nh_w = Hp // ws; nw_w = Wp // ws

        xhw  = xp.permute(0, 2, 3, 1)
        xwin = xhw.view(B, nh_w, ws, nw_w, ws, C).permute(0, 1, 3, 2, 4, 5)
        xwin = xwin.reshape(B * nh_w * nw_w, ws*ws, C)
        q    = self.q(self.norm(xwin))

        ws_ov = ws + 2*ov
        xp2   = F.pad(xp, (ov, ov, ov, ov), mode='reflect')
        xuf   = xp2.unfold(2, ws_ov, ws).unfold(3, ws_ov, ws)
        xuf   = xuf.permute(0, 2, 3, 4, 5, 1).reshape(B * nh_w * nw_w, ws_ov*ws_ov, C)
        kv    = self.kv(xuf).reshape(B * nh_w * nw_w, ws_ov*ws_ov, 2, self.nh, self.hd)
        k, v  = kv.permute(2, 0, 3, 1, 4).unbind(0)

        q    = q.reshape(B * nh_w * nw_w, ws*ws, self.nh, self.hd).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out  = (attn.softmax(-1) @ v).transpose(1, 2).reshape(B * nh_w * nw_w, ws*ws, C)
        out  = self.proj(out) + xwin

        out = out.reshape(B, nh_w, nw_w, ws, ws, C).permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(B, Hp, Wp, C).permute(0, 3, 1, 2)
        return out[:, :, :H, :W]


class HATBlock(nn.Module):
    def __init__(self, ch, ws=8, ov=2, nh=8, shift=False):
        super().__init__()
        self.ws = ws; self.ss = ws // 2 if shift else 0
        self.norm1  = nn.LayerNorm(ch)
        self.w_attn = WindowMSA(ch, ws, nh)
        self.oca    = OverlappingCrossAttn(ch, ws, ov, nh)
        self.norm2  = nn.LayerNorm(ch)
        self.mlp    = nn.Sequential(
            nn.Linear(ch, ch*2, bias=False), nn.GELU(),
            nn.Linear(ch*2, ch, bias=False),
        )
        self.alpha  = nn.Parameter(torch.tensor(0.5))

    def _part(self, x, ws):
        B, H, W, C = x.shape
        return x.view(B, H//ws, ws, W//ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws*ws, C)

    def _rev(self, x, ws, H, W, B):
        return x.reshape(B, H//ws, W//ws, ws, ws, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = min(self.ws, H, W)
        ph = (ws - H % ws) % ws; pw = (ws - W % ws) % ws
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        _, _, Hp, Wp = x.shape

        xhw = x.permute(0, 2, 3, 1)
        if self.ss > 0:
            xhw = torch.roll(xhw, (-self.ss, -self.ss), (1, 2))

        xw  = self._part(xhw, ws)
        xw  = xw + self.w_attn(self.norm1(xw))
        xw  = xw + self.mlp(self.norm2(xw))
        xhw = self._rev(xw, ws, Hp, Wp, B)

        if self.ss > 0:
            xhw = torch.roll(xhw, (self.ss, self.ss), (1, 2))
        x_win = xhw.permute(0, 3, 1, 2)[:, :, :H, :W]
        x_oca = self.oca(x[:, :, :H, :W])
        return x[:, :, :H, :W] + x_win * self.alpha + x_oca * (1 - self.alpha)


# =====================================================================
# TEFA-SR 核心模块
# =====================================================================
class FFTChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, ch // reduction)
        self.fc1 = nn.Linear(ch, hidden)
        self.fc2 = nn.Linear(hidden, ch)

    def forward(self, x):
        freq = torch.fft.rfft2(x, norm='ortho')
        stat = torch.abs(freq).mean(dim=(-2, -1))
        gate = torch.sigmoid(self.fc2(F.gelu(self.fc1(stat)))).unsqueeze(-1).unsqueeze(-1)
        return x * gate


class EdgeSpatialAttention(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch // 2, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, edge_feat):
        return self.conv(edge_feat)


class HybridFreqEdgeBlock(nn.Module):
    def __init__(self, ch: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch  = ch * dw_expand
        ffn_ch = ch * ffn_expand

        self.norm1  = LayerNorm2d(ch)
        self.pw1    = nn.Conv2d(ch, dw_ch, 1)
        self.dwconv = nn.Conv2d(dw_ch, dw_ch, 3, 1, 1, groups=dw_ch)
        self.sg     = SimpleGate()
        self.pw2    = nn.Conv2d(dw_ch // 2, ch, 1)
        self.fft_ca = FFTChannelAttention(ch)
        self.esa    = EdgeSpatialAttention(ch)
        self.beta   = nn.Parameter(torch.zeros(1, ch, 1, 1))

        self.norm2  = LayerNorm2d(ch)
        self.ffn1   = nn.Conv2d(ch, ffn_ch, 1)
        self.ffn2   = nn.Conv2d(ffn_ch // 2, ch, 1)
        self.gamma  = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x, edge_feat):
        y = self.norm1(x)
        y = self.pw1(y); y = self.dwconv(y); y = self.sg(y); y = self.pw2(y)
        y = self.fft_ca(y)
        mask = self.esa(edge_feat)
        y = y * (1.0 + mask)
        x = x + y * self.beta

        z = self.norm2(x)
        z = self.ffn1(z); z = self.sg(z); z = self.ffn2(z)
        x = x + z * self.gamma
        return x


class SobelEdge(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3) / 8.
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],  dtype=torch.float32).view(1,1,3,3) / 8.
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx*gx + gy*gy + 1e-12)


# =====================================================================
# HR 噪声感知一致性模块
# =====================================================================
class HRNoiseEstimator(nn.Module):
    def __init__(self, in_ch: int = 1, feat_ch: int = 64):
        super().__init__()
        C = feat_ch
        self.branch_local = nn.Sequential(
            nn.Conv2d(in_ch, C // 2, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(C // 2, C // 2, 3, padding=1, bias=False), nn.GELU(),
        )
        self.branch_stripe = nn.Sequential(
            nn.Conv2d(in_ch, C // 4, 3, padding=2, dilation=2, bias=False), nn.GELU(),
            nn.Conv2d(C // 4, C // 4, 3, padding=4, dilation=4, bias=False), nn.GELU(),
        )
        self.branch_horiz = nn.Sequential(
            nn.Conv2d(in_ch, C // 8, (1, 7), padding=(0, 3), bias=False), nn.GELU(),
        )
        self.branch_vert = nn.Sequential(
            nn.Conv2d(in_ch, C // 8, (7, 1), padding=(3, 0), bias=False), nn.GELU(),
        )
        total = C // 2 + C // 4 + C // 8 + C // 8
        self.fuse = nn.Sequential(
            nn.Conv2d(total, C, 1, bias=False), nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1, bias=False),
            LayerNorm2d(C),
        )

    def forward(self, hr: torch.Tensor) -> torch.Tensor:
        f1 = self.branch_local(hr)
        f2 = self.branch_stripe(hr)
        f3 = self.branch_horiz(hr)
        f4 = self.branch_vert(hr)
        return self.fuse(torch.cat([f1, f2, f3, f4], dim=1))


class LRNoiseAdapter(nn.Module):
    def __init__(self, in_ch: int = 1, feat_ch: int = 64):
        super().__init__()
        C = feat_ch
        self.branch_local = nn.Sequential(
            nn.Conv2d(in_ch, C // 2, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(C // 2, C // 2, 3, padding=1, bias=False), nn.GELU(),
        )
        self.branch_stripe = nn.Sequential(
            nn.Conv2d(in_ch, C // 4, 3, padding=2, dilation=2, bias=False), nn.GELU(),
            nn.Conv2d(C // 4, C // 4, 3, padding=4, dilation=4, bias=False), nn.GELU(),
        )
        self.branch_horiz = nn.Sequential(
            nn.Conv2d(in_ch, C // 8, (1, 7), padding=(0, 3), bias=False), nn.GELU(),
        )
        self.branch_vert = nn.Sequential(
            nn.Conv2d(in_ch, C // 8, (7, 1), padding=(3, 0), bias=False), nn.GELU(),
        )
        total = C // 2 + C // 4 + C // 8 + C // 8
        self.fuse = nn.Sequential(
            nn.Conv2d(total, C, 1, bias=False), nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1, bias=False),
            LayerNorm2d(C),
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        f1 = self.branch_local(lr)
        f2 = self.branch_stripe(lr)
        f3 = self.branch_horiz(lr)
        f4 = self.branch_vert(lr)
        return self.fuse(torch.cat([f1, f2, f3, f4], dim=1))


class NoiseStyleModulation(nn.Module):
    def __init__(self, feat_ch: int, noise_ch: int):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Conv2d(noise_ch, feat_ch, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, 1, bias=False), nn.Tanh(),
        )
        self.beta_net = nn.Sequential(
            nn.Conv2d(noise_ch, feat_ch, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, 1, bias=False), nn.Tanh(),
        )
        nn.init.zeros_(self.gamma_net[-2].weight)
        nn.init.zeros_(self.beta_net[-2].weight)

    def forward(self, feat: torch.Tensor, noise_feat: torch.Tensor) -> torch.Tensor:
        if noise_feat.shape[-2:] != feat.shape[-2:]:
            noise_feat = F.interpolate(noise_feat, size=feat.shape[-2:],
                                       mode='bilinear', align_corners=False)
        gamma = self.gamma_net(noise_feat)
        beta  = self.beta_net(noise_feat)
        return feat * (1.0 + gamma) + beta


# =====================================================================
# 融合模块
# =====================================================================
class EdgeGuidedHATBlock(nn.Module):
    def __init__(self, ch, ws=8, ov=2, nh=8, shift=False):
        super().__init__()
        self.hat  = HATBlock(ch, ws, ov, nh, shift)
        self.esa  = EdgeSpatialAttention(ch)
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, edge_feat):
        y    = self.hat(x)
        mask = self.esa(edge_feat)
        return y + y * mask * self.beta


class DeepFIU(nn.Module):
    def __init__(self, ch_ll, ch_hf):
        super().__init__()
        self.hf2ll = nn.Sequential(
            nn.Conv2d(ch_hf, ch_ll, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, ch_ll), ch_ll), nn.Sigmoid(),
        )
        self.ll2hf = nn.Sequential(
            nn.Conv2d(ch_ll, ch_hf, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, ch_hf), ch_hf), nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta  = nn.Parameter(torch.tensor(0.1))

    def forward(self, ll, hf):
        ll_out = ll + ll * self.hf2ll(hf) * self.alpha
        hf_out = hf + hf * self.ll2hf(ll) * self.beta
        return ll_out, hf_out


class DenseAggregation(nn.Module):
    def __init__(self, n_inputs, ch):
        super().__init__()
        self.proj = nn.Conv2d(n_inputs * ch, ch, 1, bias=False)

    def forward(self, feat_list):
        return self.proj(torch.cat(feat_list, dim=1))


# =====================================================================
# 主网络
# =====================================================================
class WIRSR_TEFA_Net(nn.Module):
    def __init__(self, in_ch=1, feat_ch=128, noise_feat_ch=64,
                 n_hat=8, n_hfb=8, n_fiu=4,
                 scale=4, ws=8, ov=2, nh=8):
        super().__init__()
        self.scale = scale
        C  = feat_ch
        NC = noise_feat_ch

        # ── 噪声感知模块 ──
        self.lr_noise_adapter   = LRNoiseAdapter(in_ch, NC)
        self.hr_noise_estimator = HRNoiseEstimator(in_ch, NC)

        # 噪声调制
        self.noise_mod_shallow = NoiseStyleModulation(C, NC)
        self.noise_mod_ll      = NoiseStyleModulation(C, NC)
        self.noise_mod_hf      = NoiseStyleModulation(C, NC)

        # ── 边缘编码 ──
        self.sobel    = SobelEdge()
        self.edge_enc = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1), nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1),
        )

        # ── 浅层特征提取 ──
        self.shallow = nn.Sequential(
            nn.Conv2d(in_ch, C, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1, bias=False),
        )

        # ── 小波变换 ──
        self.dwt  = HaarDWT()
        self.idwt = HaarIDWT()

        # ── LL 分支 ──
        self.ll_proj   = nn.Conv2d(C, C, 1, bias=False)
        self.ll_blocks = nn.ModuleList([
            EdgeGuidedHATBlock(C, ws, ov, nh, shift=(i % 2 == 1))
            for i in range(n_hat)
        ])
        self.ll_dense  = DenseAggregation(n_hat + 1, C)

        # ── HF 分支 ──
        self.hf_proj   = nn.Conv2d(C * 3, C, 1, bias=False)
        self.hf_blocks = nn.ModuleList([HybridFreqEdgeBlock(C) for _ in range(n_hfb)])
        self.hf_dense  = DenseAggregation(n_hfb + 1, C)
        self.hf_unproj = nn.Conv2d(C, C * 3, 1, bias=False)

        # ── 双向频率交互 ──
        self.fius = nn.ModuleList([DeepFIU(C, C * 3) for _ in range(n_fiu)])

        # ── 融合 ──
        self.fusion = nn.Sequential(
            nn.Conv2d(C * 2, C, 1, bias=False), nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1, bias=False),
        )

        # ── 上采样 ──
        self.up1 = nn.Sequential(
            nn.Conv2d(C, C * 4, 3, padding=1, bias=False), nn.GELU(),
            nn.PixelShuffle(2),
            nn.Conv2d(C, C, 3, padding=1, bias=False), nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(C, C * 4, 3, padding=1, bias=False), nn.GELU(),
            nn.PixelShuffle(2),
            nn.Conv2d(C, in_ch, 3, padding=1, bias=False),
        )
        nn.init.zeros_(self.up2[-1].weight)

    def _pad_even(self, x):
        _, _, H, W = x.shape
        ph = H % 2; pw = W % 2
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), 'reflect')
        return x, H, W

    def forward(self, lr: torch.Tensor,
                hr: torch.Tensor = None) -> tuple:
        s   = self.scale
        bic = F.interpolate(lr, scale_factor=s, mode='bicubic',
                            align_corners=False).clamp(0, 1)

        # ── 噪声感知 ──
        lr_noise_feat = self.lr_noise_adapter(lr)

        hr_noise_feat = None
        if self.training and hr is not None:
            hr_noise_feat = self.hr_noise_estimator(hr)

        # ── 边缘特征 ──
        e_feat = self.edge_enc(self.sobel(lr))

        # ── 浅层特征 + 噪声调制 ──
        f0   = self.shallow(lr)
        f0   = self.noise_mod_shallow(f0, lr_noise_feat)
        skip = f0

        # ── 小波分解 ──
        fp, H, W = self._pad_even(f0)
        wav = self.dwt(fp)
        C   = f0.shape[1]

        ll = self.ll_proj(wav[:, :C])
        hf = self.hf_proj(wav[:, C:])

        e_feat_d = F.interpolate(e_feat, size=ll.shape[-2:],
                                 mode='bilinear', align_corners=False)

        # ── LL 分支 ──
        ll_feats = [ll]
        for blk in self.ll_blocks:
            ll = blk(ll, e_feat_d)
            ll_feats.append(ll)
        ll = self.ll_dense(ll_feats)
        ll = self.noise_mod_ll(ll, lr_noise_feat)

        # ── HF 分支 ──
        hf_feats = [hf]
        for blk in self.hf_blocks:
            hf = blk(hf, e_feat_d)
            hf_feats.append(hf)
        hf = self.hf_dense(hf_feats)
        hf = self.noise_mod_hf(hf, lr_noise_feat)
        hf = self.hf_unproj(hf)

        # ── 双向交互 ──
        for fiu in self.fius:
            ll, hf = fiu(ll, hf)

        # ── 重建 ──
        wav_out  = torch.cat([ll, hf], dim=1)
        feat_wav = self.idwt(wav_out)[:, :, :H, :W]
        merged   = self.fusion(torch.cat([feat_wav, skip], dim=1))
        residual = self.up2(self.up1(merged))
        sr       = torch.clamp(bic + residual, 0., 1.)

        if self.training and hr_noise_feat is not None:
            return sr, lr_noise_feat, hr_noise_feat
        return sr