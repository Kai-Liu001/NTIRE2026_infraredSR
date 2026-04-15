import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from typing import Optional


def to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return x
    return (x, x)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def window_partition(x, window_size: int):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size: int, h: int, w: int):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def _sparse_qmk(q: torch.Tensor, k: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    b, n, d = q.shape
    topk = int(index.shape[-1])

    index = index.to(dtype=torch.long)

    b_chunk_size = 512
    topk_chunk_size = 32

    if torch.is_grad_enabled():
        out_b = []
        for b_start in range(0, b, int(b_chunk_size)):
            b_end = min(b_start + int(b_chunk_size), b)
            qb = q[b_start:b_end]
            kb = k[b_start:b_end]
            ib = index[b_start:b_end]
            bb = int(b_end - b_start)

            out_topk = []
            for start in range(0, topk, int(topk_chunk_size)):
                end = min(start + int(topk_chunk_size), topk)
                chunk = int(end - start)

                idx_chunk = ib[:, :, start:end].reshape(bb, n * chunk)
                idx_chunk = idx_chunk.unsqueeze(-1).expand(bb, n * chunk, d)
                k_sel = kb.gather(dim=1, index=idx_chunk)
                k_sel = k_sel.view(bb, n, chunk, d)
                out_topk.append((qb.unsqueeze(2) * k_sel).sum(dim=-1))
                del idx_chunk, k_sel

            out_b.append(torch.cat(out_topk, dim=-1))
            del qb, kb, ib, out_topk

        return torch.cat(out_b, dim=0)

    attn = torch.empty((b, n, topk), device=q.device, dtype=q.dtype)
    for b_start in range(0, b, int(b_chunk_size)):
        b_end = min(b_start + int(b_chunk_size), b)
        qb = q[b_start:b_end]
        kb = k[b_start:b_end]
        ib = index[b_start:b_end]
        bb = int(b_end - b_start)

        for start in range(0, topk, int(topk_chunk_size)):
            end = min(start + int(topk_chunk_size), topk)
            chunk = int(end - start)

            idx_chunk = ib[:, :, start:end].reshape(bb, n * chunk)
            idx_chunk = idx_chunk.unsqueeze(-1).expand(bb, n * chunk, d)

            k_sel = kb.gather(dim=1, index=idx_chunk)
            k_sel = k_sel.view(bb, n, chunk, d)

            attn[b_start:b_end, :, start:end] = (qb.unsqueeze(2) * k_sel).sum(dim=-1)
            del idx_chunk, k_sel

        del qb, kb, ib

    return attn


def _sparse_amv(attn: torch.Tensor, v: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    b, n, d = v.shape
    topk = int(index.shape[-1])

    index = index.to(dtype=torch.long)

    b_chunk_size = 512
    topk_chunk_size = 32

    if torch.is_grad_enabled():
        out = None
        for start in range(0, topk, int(topk_chunk_size)):
            end = min(start + int(topk_chunk_size), topk)
            chunk = int(end - start)

            partial_b = []
            for b_start in range(0, b, int(b_chunk_size)):
                b_end = min(b_start + int(b_chunk_size), b)
                vb = v[b_start:b_end]
                ib = index[b_start:b_end]
                ab = attn[b_start:b_end]
                bb = int(b_end - b_start)

                idx_chunk = ib[:, :, start:end].reshape(bb, n * chunk)
                idx_chunk = idx_chunk.unsqueeze(-1).expand(bb, n * chunk, d)
                v_sel = vb.gather(dim=1, index=idx_chunk)
                v_sel = v_sel.view(bb, n, chunk, d)

                attn_chunk = ab[:, :, start:end].unsqueeze(-1)
                partial_b.append((attn_chunk * v_sel).sum(dim=2))
                del vb, ib, ab, idx_chunk, v_sel, attn_chunk

            partial = torch.cat(partial_b, dim=0)
            del partial_b
            if out is None:
                out = partial
            else:
                out = out + partial
            del partial
        return out

    out = torch.zeros((b, n, d), device=v.device, dtype=v.dtype)
    for b_start in range(0, b, int(b_chunk_size)):
        b_end = min(b_start + int(b_chunk_size), b)
        vb = v[b_start:b_end]
        ib = index[b_start:b_end]
        ab = attn[b_start:b_end]
        outb = out[b_start:b_end]
        bb = int(b_end - b_start)

        for start in range(0, topk, int(topk_chunk_size)):
            end = min(start + int(topk_chunk_size), topk)
            chunk = int(end - start)

            idx_chunk = ib[:, :, start:end].reshape(bb, n * chunk)
            idx_chunk = idx_chunk.unsqueeze(-1).expand(bb, n * chunk, d)

            v_sel = vb.gather(dim=1, index=idx_chunk)
            v_sel = v_sel.view(bb, n, chunk, d)

            attn_chunk = ab[:, :, start:end].unsqueeze(-1)
            outb.add_((attn_chunk * v_sel).sum(dim=2))
            del idx_chunk, v_sel, attn_chunk

        del vb, ib, ab, outb

    return out


class DWConv(nn.Module):
    def __init__(self, hidden_features: int, kernel_size: int = 5):
        super().__init__()
        self.hidden_features = int(hidden_features)
        k = int(kernel_size)
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                self.hidden_features,
                self.hidden_features,
                kernel_size=k,
                stride=1,
                padding=(k - 1) // 2,
                dilation=1,
                groups=self.hidden_features,
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        kernel_size: int = 5,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = int(out_features) if out_features is not None else int(in_features)
        hidden_features = int(hidden_features) if hidden_features is not None else int(in_features)

        self.fc1 = nn.Linear(int(in_features), hidden_features)
        self.act = act_layer()
        self.dwconv = DWConv(hidden_features=hidden_features, kernel_size=int(kernel_size))
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        layer_id: int,
        window_size,
        num_heads: int,
        num_topk,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.layer_id = int(layer_id)
        self.window_size = window_size
        self.num_heads = int(num_heads)
        self.num_topk = num_topk
        self.qkv_bias = bool(qkv_bias)

        head_dim = self.dim // self.num_heads
        self.scale = head_dim**-0.5
        self.eps = 1e-20

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.proj = nn.Linear(self.dim, self.dim)
        self.softmax = nn.Softmax(dim=-1)

        self.topk = int(self.num_topk[self.layer_id])

    def forward(self, qkvp, pfa_values, pfa_indices, rpi, mask=None, shift: int = 0):
        b_, n, c4 = qkvp.shape
        c = c4 // 4

        qkvp = qkvp.reshape(b_, n, 4, self.num_heads, c // self.num_heads)
        qkvp = qkvp.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v, v_lepe = qkvp[0], qkvp[1], qkvp[2], qkvp[3]

        q = q * self.scale

        has_pfa_indices = int(pfa_indices[int(shift)].numel()) > 0
        has_pfa_values = int(pfa_values[int(shift)].numel()) > 0

        if not has_pfa_indices:
            attn = q @ k.transpose(-2, -1)

            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            if not self.training:
                attn.add_(relative_position_bias)
            else:
                attn = attn + relative_position_bias

            if int(shift) and mask is not None:
                nw = mask.shape[0]
                if not self.training:
                    attn = attn.view(b_ // nw, nw, self.num_heads, n, n)
                    attn.add_(mask.unsqueeze(1).unsqueeze(0))
                    attn = attn.view(-1, self.num_heads, n, n)
                else:
                    attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
                    attn = attn.view(-1, self.num_heads, n, n)
        else:
            topk = int(pfa_indices[int(shift)].shape[-1])
            q2 = q.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            k2 = k.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            smm_index = pfa_indices[int(shift)].view(b_ * self.num_heads, n, topk).long()
            attn = _sparse_qmk(q2, k2, smm_index).view(b_, self.num_heads, n, topk)

            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            relative_position_bias = relative_position_bias.expand(b_, self.num_heads, n, n)
            relative_position_bias = torch.gather(
                relative_position_bias,
                dim=-1,
                index=pfa_indices[int(shift)].to(dtype=torch.long),
            )
            if not self.training:
                attn.add_(relative_position_bias)
            else:
                attn = attn + relative_position_bias

        if not self.training:
            attn = torch.softmax(attn, dim=-1, out=attn)
        else:
            attn = self.softmax(attn)

        if has_pfa_values:
            if not self.training:
                attn.mul_(pfa_values[int(shift)])
                attn.add_(self.eps)
                denom = attn.sum(dim=-1, keepdim=True)
                denom.add_(self.eps)
                attn.div_(denom)
            else:
                attn = attn * pfa_values[int(shift)]
                denom = attn.sum(dim=-1, keepdim=True)
                attn = (attn + self.eps) / (denom + self.eps)

        if self.topk < self.window_size[0] * self.window_size[1]:
            topk_values, topk_indices = torch.topk(attn, self.topk, dim=-1, largest=True, sorted=False)
            attn = topk_values
            if has_pfa_indices:
                pfa_indices[int(shift)] = torch.gather(pfa_indices[int(shift)], dim=-1, index=topk_indices)
            else:
                pfa_indices[int(shift)] = topk_indices

        pfa_values[int(shift)] = attn

        has_pfa_indices = int(pfa_indices[int(shift)].numel()) > 0
        if not has_pfa_indices:
            x = ((attn @ v) + v_lepe).transpose(1, 2).reshape(b_, n, c)
        else:
            topk = int(pfa_indices[int(shift)].shape[-1])
            attn2 = attn.view(b_ * self.num_heads, n, topk)
            v2 = v.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            smm_index = pfa_indices[int(shift)].view(b_ * self.num_heads, n, topk).long()
            x = (_sparse_amv(attn2, v2, smm_index).view(b_, self.num_heads, n, c // self.num_heads) + v_lepe)
            x = x.transpose(1, 2).reshape(b_, n, c)

        if not self.training:
            if 'q2' in locals():
                del q2
            if 'k2' in locals():
                del k2
            if 'smm_index' in locals():
                del smm_index
            if 'attn2' in locals():
                del attn2
            if 'v2' in locals():
                del v2
            del q, k, v
            if 'relative_position_bias' in locals():
                del relative_position_bias
            if x.is_cuda:
                torch.cuda.empty_cache()

        x = self.proj(x)
        return x, pfa_values, pfa_indices

    def forward_inference(self, qkvp, pfa_values, pfa_indices, rpi, mask=None, shift: int = 0, window_chunk_size: int = 8):
        b_, n, c4 = qkvp.shape
        c = c4 // 4
        head_dim = c // self.num_heads

        pfa_indices_in = pfa_indices[int(shift)]
        pfa_values_in = pfa_values[int(shift)]

        has_pfa_indices = int(pfa_indices_in.numel()) > 0
        has_pfa_values = int(pfa_values_in.numel()) > 0

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().to(device=qkvp.device, dtype=qkvp.dtype)

        out = torch.empty((b_, n, c), device=qkvp.device, dtype=qkvp.dtype)

        if self.topk < n:
            pfa_values_out = torch.empty((b_, self.num_heads, n, self.topk), device="cpu", dtype=qkvp.dtype)
            pfa_indices_out = torch.empty((b_, self.num_heads, n, self.topk), device="cpu", dtype=torch.long)
        else:
            pfa_values_out = torch.empty(0, device="cpu", dtype=qkvp.dtype)
            pfa_indices_out = torch.empty(0, device="cpu", dtype=torch.long)

        if int(shift) and mask is not None:
            nw = int(mask.shape[0])
        else:
            nw = 0

        for start in range(0, b_, int(window_chunk_size)):
            end = min(start + int(window_chunk_size), b_)
            chunk = int(end - start)

            qkvp_chunk = qkvp[start:end]
            qkvp_chunk = qkvp_chunk.reshape(chunk, n, 4, self.num_heads, head_dim)
            qkvp_chunk = qkvp_chunk.permute(2, 0, 3, 1, 4).contiguous()
            q, k, v, v_lepe = qkvp_chunk[0], qkvp_chunk[1], qkvp_chunk[2], qkvp_chunk[3]
            q = q * self.scale
            del qkvp_chunk

            if not has_pfa_indices:
                attn = q @ k.transpose(-2, -1)
                attn.add_(relative_position_bias.unsqueeze(0))

                if int(shift) and mask is not None:
                    if mask.is_cuda:
                        idx = (torch.arange(start, end, device=mask.device) % nw).to(dtype=torch.long)
                        mask_chunk = mask[idx]
                    else:
                        idx = (torch.arange(start, end) % nw).to(dtype=torch.long)
                        mask_chunk = mask[idx].to(device=qkvp.device, dtype=qkvp.dtype)
                    attn.add_(mask_chunk.unsqueeze(1))
                    del idx, mask_chunk

                attn = torch.softmax(attn, dim=-1, out=attn)
            else:
                pfa_indices_chunk = pfa_indices_in[start:end]
                if pfa_indices_chunk.is_cuda:
                    pfa_indices_chunk = pfa_indices_chunk.to(dtype=torch.long)
                else:
                    pfa_indices_chunk = pfa_indices_chunk.to(device=qkvp.device, dtype=torch.long)
                topk_in = int(pfa_indices_chunk.shape[-1])

                q2 = q.contiguous().view(chunk * self.num_heads, n, head_dim)
                k2 = k.contiguous().view(chunk * self.num_heads, n, head_dim)
                smm_index = pfa_indices_chunk.view(chunk * self.num_heads, n, topk_in).long()
                attn = _sparse_qmk(q2, k2, smm_index).view(chunk, self.num_heads, n, topk_in)
                del q2, k2, smm_index

                rpb = relative_position_bias.unsqueeze(0).expand(chunk, -1, -1, -1)
                rpb = torch.gather(rpb, dim=-1, index=pfa_indices_chunk)
                attn.add_(rpb)
                del rpb

                attn = torch.softmax(attn, dim=-1, out=attn)

            if has_pfa_values:
                pfa_values_chunk = pfa_values_in[start:end]
                if pfa_values_chunk.is_cuda:
                    pfa_values_chunk = pfa_values_chunk.to(dtype=qkvp.dtype)
                else:
                    pfa_values_chunk = pfa_values_chunk.to(device=qkvp.device, dtype=qkvp.dtype)

                attn.mul_(pfa_values_chunk)
                attn.add_(self.eps)
                denom = attn.sum(dim=-1, keepdim=True)
                denom.add_(self.eps)
                attn.div_(denom)
                del pfa_values_chunk, denom

            if self.topk < n:
                topk_values, topk_indices = torch.topk(attn, self.topk, dim=-1, largest=True, sorted=False)
                attn = topk_values
                if has_pfa_indices:
                    pfa_indices_chunk_out = torch.gather(pfa_indices_chunk, dim=-1, index=topk_indices)
                else:
                    pfa_indices_chunk_out = topk_indices
                del topk_indices

                pfa_values_out[start:end] = attn.to(device="cpu")
                pfa_indices_out[start:end] = pfa_indices_chunk_out.to(device="cpu", dtype=torch.long)

                topk = int(self.topk)
                attn2 = attn.view(chunk * self.num_heads, n, topk)
                v2 = v.contiguous().view(chunk * self.num_heads, n, head_dim)
                smm_index2 = pfa_indices_chunk_out.view(chunk * self.num_heads, n, topk).long()
                x = (_sparse_amv(attn2, v2, smm_index2).view(chunk, self.num_heads, n, head_dim) + v_lepe)
                x = x.transpose(1, 2).reshape(chunk, n, c)
                del attn2, v2, smm_index2, pfa_indices_chunk_out
            else:
                x = ((attn @ v) + v_lepe).transpose(1, 2).reshape(chunk, n, c)

            del attn, q, k, v, v_lepe

            out[start:end] = self.proj(x)
            del x

            if out.is_cuda:
                torch.cuda.empty_cache()

        if self.topk < n:
            pfa_values[int(shift)] = pfa_values_out
            pfa_indices[int(shift)] = pfa_indices_out
        else:
            pfa_values[int(shift)] = pfa_values_out
            pfa_indices[int(shift)] = pfa_indices_out

        del relative_position_bias
        return out, pfa_values, pfa_indices


class PFTransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        block_id: int,
        layer_id: int,
        input_resolution,
        num_heads: int,
        num_topk,
        window_size: int,
        shift_size: int,
        convffn_kernel_size: int,
        mlp_ratio: float,
        qkv_bias: bool = True,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.dim = int(dim)
        self.layer_id = int(layer_id)
        self.input_resolution = input_resolution
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.mlp_ratio = float(mlp_ratio)
        self.convffn_kernel_size = int(convffn_kernel_size)

        self.norm1 = norm_layer(self.dim)
        self.norm2 = norm_layer(self.dim)

        self.wqkv = nn.Linear(self.dim, 3 * self.dim, bias=bool(qkv_bias))

        self.v_LePE = DWConv(hidden_features=self.dim, kernel_size=self.convffn_kernel_size)

        self.attn_win = WindowAttention(
            self.dim,
            layer_id=int(layer_id),
            window_size=to_2tuple(self.window_size),
            num_heads=self.num_heads,
            num_topk=num_topk,
            qkv_bias=bool(qkv_bias),
        )

        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.convffn = ConvFFN(
            in_features=self.dim,
            hidden_features=mlp_hidden_dim,
            kernel_size=self.convffn_kernel_size,
            act_layer=act_layer,
        )

    def forward(self, x, pfa_list, x_size, params):
        pfa_values, pfa_indices = pfa_list[0], pfa_list[1]
        h, w = x_size
        b, n, c = x.shape
        c4 = 4 * c

        shortcut = x
        x = self.norm1(x)
        x_qkv = self.wqkv(x)

        v_lepe = self.v_LePE(torch.split(x_qkv, c, dim=-1)[-1], x_size)
        x_qkvp = torch.cat([x_qkv, v_lepe], dim=-1)

        if self.shift_size > 0:
            shift = 1
            shifted_x = torch.roll(
                x_qkvp.reshape(b, h, w, c4),
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shift = 0
            shifted_x = x_qkvp.reshape(b, h, w, c4)

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c4)

        attn_windows, pfa_values, pfa_indices = self.attn_win(
            x_windows,
            pfa_values=pfa_values,
            pfa_indices=pfa_indices,
            rpi=params["rpi_sa"],
            mask=params["attn_mask"],
            shift=shift,
        )

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x

        x_win = attn_x

        x = shortcut + x_win.view(b, n, c)
        x = x + self.convffn(self.norm2(x), x_size)

        pfa_list = [pfa_values, pfa_indices]
        return x, pfa_list

    def forward_inference(self, x, pfa_list, x_size, params):
        pfa_values, pfa_indices = pfa_list[0], pfa_list[1]
        h, w = x_size
        b, n, c = x.shape
        c4 = 4 * c

        x_norm1 = self.norm1(x)
        x_qkv = self.wqkv(x_norm1)
        v = x_qkv[..., 2 * c :]
        v_lepe = self.v_LePE(v, x_size)
        x_qkvp = torch.cat([x_qkv, v_lepe], dim=-1)
        del x_norm1, x_qkv, v, v_lepe

        if self.shift_size > 0:
            shift = 1
            shifted_x = torch.roll(
                x_qkvp.reshape(b, h, w, c4),
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shift = 0
            shifted_x = x_qkvp.reshape(b, h, w, c4)

        del x_qkvp

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c4)
        del shifted_x

        attn_windows, pfa_values, pfa_indices = self.attn_win.forward_inference(
            x_windows,
            pfa_values=pfa_values,
            pfa_indices=pfa_indices,
            rpi=params["rpi_sa"],
            mask=params["attn_mask"],
            shift=shift,
        )
        del x_windows

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        del attn_windows

        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        del shifted_x

        x.add_(attn_x.view(b, n, c))
        del attn_x

        x_norm2 = self.norm2(x)
        x_ffn = self.convffn(x_norm2, x_size)
        del x_norm2
        x.add_(x_ffn)
        del x_ffn

        pfa_list = [pfa_values, pfa_indices]
        return x, pfa_list


class BasicBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution,
        idx: int,
        layer_id: int,
        depth: int,
        num_heads: int,
        num_topk,
        window_size: int,
        convffn_kernel_size: int,
        mlp_ratio: float,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.dim = int(dim)
        self.input_resolution = input_resolution
        self.depth = int(depth)
        self.use_checkpoint = bool(use_checkpoint)

        self.layers = nn.ModuleList()
        for i in range(self.depth):
            self.layers.append(
                PFTransformerLayer(
                    dim=self.dim,
                    block_id=int(idx),
                    layer_id=int(layer_id) + i,
                    input_resolution=input_resolution,
                    num_heads=int(num_heads),
                    num_topk=num_topk,
                    window_size=int(window_size),
                    shift_size=0 if (i % 2 == 0) else int(window_size) // 2,
                    convffn_kernel_size=int(convffn_kernel_size),
                    mlp_ratio=float(mlp_ratio),
                    qkv_bias=bool(qkv_bias),
                    norm_layer=norm_layer,
                )
            )

        self.downsample = downsample(input_resolution, dim=self.dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, pfa_list, x_size, params):
        pfa_values, pfa_indices = pfa_list[0], pfa_list[1]
        pv0, pv1 = pfa_values[0], pfa_values[1]
        pi0, pi1 = pfa_indices[0], pfa_indices[1]

        for layer in self.layers:
            if self.use_checkpoint and x.requires_grad:
                def _layer_forward(_x, _pv0, _pv1, _pi0, _pi1, _attn_mask, _rpi_sa, _layer=layer):
                    _pfa_list = [[_pv0, _pv1], [_pi0, _pi1]]
                    _params = {'attn_mask': _attn_mask, 'rpi_sa': _rpi_sa}
                    _x, _pfa_list = _layer(_x, _pfa_list, x_size, _params)
                    _pv0, _pv1 = _pfa_list[0][0], _pfa_list[0][1]
                    _pi0, _pi1 = _pfa_list[1][0], _pfa_list[1][1]
                    return _x, _pv0, _pv1, _pi0, _pi1

                x, pv0, pv1, pi0, pi1 = checkpoint.checkpoint(
                    _layer_forward,
                    x,
                    pv0,
                    pv1,
                    pi0,
                    pi1,
                    params['attn_mask'],
                    params['rpi_sa'],
                )
            else:
                _pfa_list = [[pv0, pv1], [pi0, pi1]]
                x, _pfa_list = layer(x, _pfa_list, x_size, params)
                pv0, pv1 = _pfa_list[0][0], _pfa_list[0][1]
                pi0, pi1 = _pfa_list[1][0], _pfa_list[1][1]

        pfa_list = [[pv0, pv1], [pi0, pi1]]

        if self.downsample is not None:
            x = self.downsample(x)
        return x, pfa_list

    def forward_inference(self, x, pfa_list, x_size, attn_mask_cpu, rpi_sa):
        pfa_values, pfa_indices = pfa_list[0], pfa_list[1]
        pv0, pv1 = pfa_values[0], pfa_values[1]
        pi0, pi1 = pfa_indices[0], pfa_indices[1]

        for layer in self.layers:
            need_mask = int(layer.shift_size) > 0 and int(pi1.numel()) == 0
            if need_mask:
                attn_mask = attn_mask_cpu
            else:
                attn_mask = None

            params = {"attn_mask": attn_mask, "rpi_sa": rpi_sa}

            x_prev = x
            _pfa_list = [[pv0, pv1], [pi0, pi1]]
            x, _pfa_list = layer.forward_inference(x_prev, _pfa_list, x_size, params)
            pv0, pv1 = _pfa_list[0][0], _pfa_list[0][1]
            pi0, pi1 = _pfa_list[1][0], _pfa_list[1][1]

            del x_prev, _pfa_list, params, attn_mask
            if x.is_cuda:
                torch.cuda.empty_cache()

        pfa_list = [[pv0, pv1], [pi0, pi1]]

        if self.downsample is not None:
            x = self.downsample(x)
        return x, pfa_list


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = int(in_chans)
        self.embed_dim = int(embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = int(in_chans)
        self.embed_dim = int(embed_dim)

    def forward(self, x: torch.Tensor, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x


class PFTB(nn.Module):
    def __init__(
        self,
        dim: int,
        idx: int,
        layer_id: int,
        input_resolution,
        depth: int,
        num_heads: int,
        num_topk,
        window_size: int,
        convffn_kernel_size: int,
        mlp_ratio: float,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint: bool = False,
        img_size=224,
        patch_size=4,
        resi_connection: str = "1conv",
    ):
        super().__init__()

        self.dim = int(dim)
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=self.dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=self.dim, norm_layer=None
        )

        self.residual_group = BasicBlock(
            dim=self.dim,
            input_resolution=input_resolution,
            idx=int(idx),
            layer_id=int(layer_id),
            depth=int(depth),
            num_heads=int(num_heads),
            num_topk=num_topk,
            window_size=int(window_size),
            convffn_kernel_size=int(convffn_kernel_size),
            mlp_ratio=float(mlp_ratio),
            qkv_bias=bool(qkv_bias),
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=bool(use_checkpoint),
        )

        if str(resi_connection) == "1conv":
            self.conv = nn.Conv2d(self.dim, self.dim, 3, 1, 1)
        elif str(resi_connection) == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(self.dim, self.dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(self.dim // 4, self.dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(self.dim // 4, self.dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"Unsupported resi_connection: {resi_connection}")

    def forward(self, x, pfa_list, x_size, params):
        x_blk, pfa_list = self.residual_group(x, pfa_list, x_size, params)
        return self.patch_embed(self.conv(self.patch_unembed(x_blk, x_size))) + x, pfa_list

    def forward_inference(self, x, pfa_list, x_size, attn_mask_cpu, rpi_sa):
        x_blk, pfa_list = self.residual_group.forward_inference(x, pfa_list, x_size, attn_mask_cpu, rpi_sa)

        res_img = self.patch_unembed(x_blk, x_size)
        del x_blk
        if res_img.is_cuda:
            torch.cuda.empty_cache()

        res_img = self.conv(res_img)
        res = self.patch_embed(res_img)
        del res_img

        x.add_(res)
        del res
        if x.is_cuda:
            torch.cuda.empty_cache()
        return x, pfa_list


class Upsample(nn.Sequential):
    def __init__(self, scale: int, num_feat: int):
        m = []
        sc = int(scale)
        nf = int(num_feat)
        if (sc & (sc - 1)) == 0:
            for _ in range(int(math.log(sc, 2))):
                m.append(nn.Conv2d(nf, 4 * nf, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif sc == 3:
            m.append(nn.Conv2d(nf, 9 * nf, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"scale {sc} is not supported")
        super().__init__(*m)


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale: int, num_feat: int, num_out_ch: int, input_resolution=None):
        m = []
        sc = int(scale)
        nf = int(num_feat)
        noc = int(num_out_ch)
        m.append(nn.Conv2d(nf, (sc**2) * noc, 3, 1, 1))
        m.append(nn.PixelShuffle(sc))
        super().__init__(*m)


class PFT(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=1,
        embed_dim=240,
        depths=(4, 4, 4, 6, 6, 6),
        num_heads=6,
        num_topk=None,
        window_size=32,
        convffn_kernel_size=7,
        mlp_ratio=2.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=4,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
        **kwargs,
    ):
        super().__init__()

        if num_topk is None:
            total_layers = int(sum(list(depths)))
            num_topk = [window_size * window_size for _ in range(total_layers)]

        in_chans = int(in_chans)
        internal_ch = 3 if in_chans == 1 else in_chans
        num_in_ch = int(internal_ch)
        num_out_ch = int(internal_ch)
        num_feat = 64

        self.img_range = float(img_range)

        if int(internal_ch) == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.upscale = int(upscale)
        self.upsampler = str(upsampler)

        self.conv_first = nn.Conv2d(num_in_ch, int(embed_dim), 3, 1, 1)

        self.num_layers = len(depths)
        self.layer_id = 0
        self.embed_dim = int(embed_dim)
        self.ape = bool(ape)
        self.patch_norm = bool(patch_norm)
        self.num_features = int(embed_dim)
        self.mlp_ratio = float(mlp_ratio)
        self.window_size = int(window_size)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=int(embed_dim),
            embed_dim=int(embed_dim),
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=int(embed_dim),
            embed_dim=int(embed_dim),
            norm_layer=norm_layer if self.patch_norm else None,
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, int(embed_dim)))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        relative_position_index_sa = self.calculate_rpi_sa()
        self.register_buffer("relative_position_index_SA", relative_position_index_sa)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = PFTB(
                dim=int(embed_dim),
                idx=int(i_layer),
                layer_id=int(self.layer_id),
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=int(depths[i_layer]),
                num_heads=int(num_heads),
                num_topk=num_topk,
                window_size=int(window_size),
                convffn_kernel_size=int(convffn_kernel_size),
                mlp_ratio=self.mlp_ratio,
                qkv_bias=bool(qkv_bias),
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=bool(use_checkpoint),
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=str(resi_connection),
            )
            self.layers.append(layer)
            self.layer_id = self.layer_id + int(depths[i_layer])

        self.norm = norm_layer(self.num_features)

        if str(resi_connection) == "1conv":
            self.conv_after_body = nn.Conv2d(int(embed_dim), int(embed_dim), 3, 1, 1)
        elif str(resi_connection) == "3conv":
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(int(embed_dim), int(embed_dim) // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(int(embed_dim) // 4, int(embed_dim) // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(int(embed_dim) // 4, int(embed_dim), 3, 1, 1),
            )
        else:
            raise ValueError(f"Unsupported resi_connection: {resi_connection}")

        if self.upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(int(embed_dim), num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.upsample = Upsample(self.upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        elif self.upsampler == "pixelshuffledirect":
            self.upsample = UpsampleOneStep(self.upscale, int(embed_dim), num_out_ch, (patches_resolution[0], patches_resolution[1]))

        elif self.upsampler == "nearest+conv":
            if int(self.upscale) != 4:
                raise ValueError("nearest+conv only supports x4")
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(int(embed_dim), num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        else:
            self.conv_last = nn.Conv2d(int(embed_dim), num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, params):
        x_size = (int(x.shape[2]), int(x.shape[3]))

        pfa_values = [x.new_empty(0), x.new_empty(0)]
        pfa_indices = [x.new_empty(0, dtype=torch.long), x.new_empty(0, dtype=torch.long)]
        pfa_list = [pfa_values, pfa_indices]

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        for layer in self.layers:
            x, pfa_list = layer(x, pfa_list, x_size, params)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward_features_inference(self, x, attn_mask_cpu, rpi_sa):
        x_size = (int(x.shape[2]), int(x.shape[3]))

        pfa_values = [x.new_empty(0), x.new_empty(0)]
        pfa_indices = [x.new_empty(0, dtype=torch.long), x.new_empty(0, dtype=torch.long)]
        pfa_list = [pfa_values, pfa_indices]

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        for layer in self.layers:
            x_prev = x
            x, pfa_list = layer.forward_inference(x_prev, pfa_list, x_size, attn_mask_cpu, rpi_sa)
            del x_prev
            if x.is_cuda:
                torch.cuda.empty_cache()

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        h, w = int(x_size[0]), int(x_size[1])
        img_mask = torch.zeros((1, h, w, 1))

        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -(self.window_size // 2)),
            slice(-(self.window_size // 2), None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -(self.window_size // 2)),
            slice(-(self.window_size // 2), None),
        )

        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def _forward_once(self, x: torch.Tensor):
        if int(x.shape[1]) == 1 and int(self.mean.shape[1]) == 3:
            x = x.repeat(1, 3, 1, 1)

        h_ori, w_ori = int(x.size(-2)), int(x.size(-1))

        mod = int(self.window_size)
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori

        h = h_ori + int(h_pad)
        w = w_ori + int(w_pad)

        if int(h_pad) > 0:
            x = torch.cat([x, torch.flip(x[:, :, -int(h_pad) :, :], [2])], dim=2)
        x = x[:, :, :h, :]
        if int(w_pad) > 0:
            x = torch.cat([x, torch.flip(x[:, :, :, -int(w_pad) :], [3])], dim=3)
        x = x[:, :, :, :w]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {"attn_mask": attn_mask, "rpi_sa": self.relative_position_index_SA}

        if self.upsampler == "pixelshuffle":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == "pixelshuffledirect":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.upsample(x)

        elif self.upsampler == "nearest+conv":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode="nearest")))
            x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode="nearest")))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))

        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, params)) + x_first
            x = x + self.conv_last(res)
            del x_first, res

        x = x / self.img_range + self.mean
        x = x[..., : h_ori * self.upscale, : w_ori * self.upscale]

        if int(x.shape[1]) == 3:
            x = x.mean(dim=1, keepdim=True)
        return x

    def forward(self, x):
        outputs_list = []

        for k in range(4):
            for do_hflip in (False, True):
                x_t = torch.rot90(x, k=k, dims=(-2, -1))
                if do_hflip:
                    x_t = torch.flip(x_t, dims=(-1,))

                out_t = self._forward_once(x_t)

                if do_hflip:
                    out_t = torch.flip(out_t, dims=(-1,))
                out_t = torch.rot90(out_t, k=-k, dims=(-2, -1))

                outputs_list.append(out_t.float())

        out = torch.stack(outputs_list, dim=0).mean(dim=0)

        return out.float()

    @torch.inference_mode()
    def forward_inference(self, x: torch.Tensor):
        if int(x.shape[1]) == 1 and int(self.mean.shape[1]) == 3:
            x = x.repeat(1, 3, 1, 1)

        h_ori, w_ori = int(x.size(-2)), int(x.size(-1))

        mod = int(self.window_size)
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori

        h = h_ori + int(h_pad)
        w = w_ori + int(w_pad)

        if int(h_pad) > 0:
            x = torch.cat([x, torch.flip(x[:, :, -int(h_pad) :, :], [2])], dim=2)
        x = x[:, :, :h, :]
        if int(w_pad) > 0:
            x = torch.cat([x, torch.flip(x[:, :, :, -int(w_pad) :], [3])], dim=3)
        x = x[:, :, :, :w]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        attn_mask = self.calculate_mask([h, w]).to(dtype=torch.float16)
        rpi_sa = self.relative_position_index_SA

        if self.upsampler == "pixelshuffle":
            x = self.conv_first(x)
            body = self.forward_features_inference(x, attn_mask, rpi_sa)
            x = self.conv_after_body(body) + x
            del body
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == "pixelshuffledirect":
            x = self.conv_first(x)
            body = self.forward_features_inference(x, attn_mask, rpi_sa)
            x = self.conv_after_body(body) + x
            del body
            x = self.upsample(x)

        elif self.upsampler == "nearest+conv":
            x = self.conv_first(x)
            body = self.forward_features_inference(x, attn_mask, rpi_sa)
            x = self.conv_after_body(body) + x
            del body
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode="nearest")))
            x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode="nearest")))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))

        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features_inference(x_first, attn_mask, rpi_sa)) + x_first
            x = x + self.conv_last(res)
            del x_first, res

        del attn_mask, rpi_sa
        if x.is_cuda:
            torch.cuda.empty_cache()

        x = x / self.img_range + self.mean
        x = x[..., : h_ori * self.upscale, : w_ori * self.upscale]

        if int(x.shape[1]) == 3:
            x = x.mean(dim=1, keepdim=True)
        return x

def pft_x4():
    inp_channels = 3
    out_channels = 3

    img_size = 64
    patch_size = 1
    embed_dim = 240
    depths = (4, 4, 4, 6, 6, 6)
    num_heads = 6
    num_topk = [
        1024, 1024, 1024, 1024,
        256, 256, 256, 256,
        128, 128, 128, 128,
        64, 64, 64, 64, 64, 64,
        32, 32, 32, 32, 32, 32,
        16, 16, 16, 16, 16, 16,
    ]
    window_size = 32
    convffn_kernel_size = 7
    mlp_ratio = 2.0
    img_range = 1.0
    upsampler = 'pixelshuffle'
    resi_connection = '1conv'

    if inp_channels != out_channels:
        raise ValueError(f'PFT requires inp_channels == out_channels, got {inp_channels} vs {out_channels}')

    return PFT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=inp_channels,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        num_topk=num_topk,
        window_size=window_size,
        convffn_kernel_size=convffn_kernel_size,
        mlp_ratio=mlp_ratio,
        upscale=4,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
    )
