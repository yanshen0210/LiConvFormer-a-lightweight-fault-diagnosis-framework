from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1,
                 padding=None, use_norm=True, use_act=True):
        super().__init__()
        block = []
        padding = padding or kernel_size // 2
        block.append(nn.Conv1d(
            in_channel, out_channel, kernel_size, stride, padding=padding, groups=groups, bias=False
        ))
        if use_norm:
            block.append(nn.BatchNorm1d(out_channel))
        if use_act:
            block.append(nn.GELU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.layernorm(x)
        return x.transpose(-1, -2)


class Add(nn.Module):
    def __init__(self, epsilon=1e-12):
        super(Add, self).__init__()
        self.epsilon = epsilon
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_relu = nn.ReLU()

    def forward(self, x):
        w = self.w_relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)

        return weight[0] * x[0] + weight[1] * x[1]


class Embedding(nn.Module):
    def __init__(self, d_in, d_out, stride=2, n=4):
        super(Embedding, self).__init__()
        d_hidden = d_out//n
        self.conv1 = nn.Conv1d(d_in, d_hidden, 1, 1)
        self.sconv = nn.ModuleList([
            nn.Conv1d(d_hidden, d_hidden, 2*i+2*stride-1,
                      stride=stride, padding=stride+i-1, groups=d_hidden, bias=False)
            for i in range(n)])
        self.act_bn = nn.Sequential(
            nn.BatchNorm1d(d_out), nn.GELU())

    def forward(self, x):
        signals = []
        x = self.conv1(x)
        for sconv in self.sconv:
            signals.append(sconv(x))
        x = torch.cat(signals, dim=1)
        return self.act_bn(x)


class BroadcastAttention(nn.Module):
    def __init__(self,
                 dim,
                 proj_drop=0.,
                 attn_drop=0.,
                 qkv_bias=True
                 ):
        super().__init__()
        self.dim = dim

        self.qkv_proj = nn.Conv1d(dim, 1 + 2 * dim, kernel_size=1, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # [B, C, N] -> [B, 1+2C, N]
        qkv = self.qkv_proj(x)

        # Query --> [B, 1, N]
        # value, key --> [B, C, N]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.dim, self.dim], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, C, N] x [B, 1, N] -> [B, C, N]
        context_vector = key * context_scores
        # [B, C, N] --> [B, C, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, C, N] * [B, C, 1] --> [B, C, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class BA_FFN_Block(nn.Module):
    def __init__(self,
                 dim,
                 ffn_dim,
                 drop=0.,
                 attn_drop=0.
                 ):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.add1 = Add()
        self.attn = BroadcastAttention(dim=dim,
                                       attn_drop=attn_drop,
                                       proj_drop=drop)

        self.norm2 = LayerNorm(dim)
        self.add2 = Add()
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, ffn_dim, 1, 1, bias=True),
            nn.GELU(),
            nn.Dropout(p=drop),
            nn.Conv1d(ffn_dim, dim, 1, 1, bias=True),
            nn.Dropout(p=drop)
        )

    def forward(self, x):
        x = self.add1([self.attn(self.norm1(x)), x])
        x = self.add2([self.ffn(self.norm2(x)), x])
        return x


class LFEL(nn.Module):
    def __init__(self, d_in, d_out, drop):
        super(LFEL, self).__init__()

        self.embed = Embedding(d_in, d_out, stride=2, n=4)
        self.block = BA_FFN_Block(dim=d_out,
                                  ffn_dim=d_out//4,
                                  drop=drop,
                                  attn_drop=drop)

    def forward(self, x):
        x = self.embed(x)
        return self.block(x)


class Liconvformer(nn.Module):
    def __init__(self, _, in_channel, out_channel, drop=0.1, dim=32):
        super(Liconvformer, self).__init__()

        self.in_layer = nn.Sequential(
            nn.AvgPool1d(2, 2),
            ConvBNReLU(in_channel, dim, kernel_size=15, stride=2)
        )

        self.LFELs = nn.Sequential(
            LFEL(dim, 2*dim, drop),
            LFEL(2*dim, 4*dim, drop),
            LFEL(4*dim, 8*dim, drop),
            nn.AdaptiveAvgPool1d(1)
        )

        self.out_layer = nn.Linear(8*dim, out_channel)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.LFELs(x)
        x = self.out_layer(x.squeeze())
        return x
