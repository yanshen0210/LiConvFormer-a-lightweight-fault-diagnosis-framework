from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


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
    def __init__(self, d_in, d_out):
        super(Embedding, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d_in, d_out // 4, i * 2 + 7,
                      stride=4, padding=i + 3, bias=False)
            for i in range(4)])
        self.act_bn = nn.Sequential(
            nn.BatchNorm1d(d_out), nn.GELU())

    def forward(self, x):
        signals = []
        for conv in self.convs:
            signals.append(conv(x))

        return self.act_bn(torch.cat(signals, dim=1))


class projector(nn.Module):
    def __init__(self, heads, dim):
        super(projector, self).__init__()
        self.q_k_v = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3,
                          stride=1 if i == 0 else 2,
                          padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, 1, 1, 0),
                nn.GELU()
            )
            for i in range(3)])
        self.MHSA = MHSA(dim, heads)
        self.add = Add()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        b, c, l = x.size()
        maps = x.view(-1, c, int(l ** 0.5), int(l ** 0.5))
        MHSA = self.MHSA(
            self.q_k_v[0](maps),
            self.q_k_v[1](maps),
            self.q_k_v[2](maps))

        return self.bn(self.add([MHSA, x]))


class MHSA(nn.Module):
    def __init__(self, emb_dim, heads):
        super(MHSA, self).__init__()
        self.dim, self.heads = emb_dim, heads

    def forward(self, q, k, v):
        q = torch.flatten(q, 2).transpose(1, 2)
        k = torch.flatten(k, 2).transpose(1, 2)
        v = torch.flatten(v, 2).transpose(1, 2)

        if self.heads == 1:
            q, k = F.softmax(q, dim=2), F.softmax(k, dim=1)
            return q.bmm(k.transpose(2, 1).bmm(v)).transpose(1, 2)
        else:
            q = q.split(self.dim // self.heads, dim=2)
            k = k.split(self.dim // self.heads, dim=2)
            v = v.split(self.dim // self.heads, dim=2)
            atts = []
            for i in range(self.heads):
                att = F.softmax(q[i], dim=2).bmm(F.softmax(k[i], dim=1).transpose(2, 1).bmm(v[i]))
                atts.append(att.transpose(1, 2))
            return torch.cat(atts, dim=1)


class FFN(nn.Module):
    def __init__(self, dim, ratio=4):
        super(FFN, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(dim, dim // ratio), nn.GELU(),
            nn.Linear(dim // ratio, dim), nn.GELU(), )
        self.add = Add()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        feature = self.MLP(x.transpose(1, 2))
        return self.bn(self.add([feature.transpose(1, 2), x]))


class CLFormer_block(nn.Module):
    def __init__(self, d_in, d_out, heads=1, blocks=1):
        super(CLFormer_block, self).__init__()

        self.embed, self.block = Embedding(d_in, d_out), nn.Sequential()
        for i in range(blocks):
            self.block.add_module(
                " block_ " + str(i), nn.Sequential(projector(heads, d_out), FFN(d_out))
            )

    def forward(self, x):
        x = self.embed(x)
        return self.block(x)


'''
" d_out " refers to the number of categories of final model output results .
" d_out " was set as 7 in this paper (6 types of faults plus 1 type of Health ).
'''
class CLFormer(nn.Module):
    def __init__(self, _, in_channel, out_channel):
        super(CLFormer, self).__init__()
        self.Encoder = nn.Sequential(
            CLFormer_block(in_channel, 4),  # B, 1, 1024 -> b, 4, 256
            CLFormer_block(4, 8),  # B, 4, 256 -> b, 8, 64
            CLFormer_block(8, 16),  # B, 8, 64 -> b, 16, 16
            nn.AdaptiveAvgPool1d(1)  # B, 16 , 16 -> b, 16 , 1
        )

        self.head_input = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.GELU(), nn.Dropout())
        self.head_output = nn.Linear(32, out_channel)

        self.zero_last_layer_weight()

    def zero_last_layer_weight(self):
        self.head_output.weight.data = torch.zeros_like(self.head_output.weight)
        self.head_output.bias.data = torch.zeros_like(self.head_output.bias)

    def forward(self, signal):
        feature = self.Encoder(signal)
        feature = self.head_input(feature.squeeze(2))
        return self.head_output(feature)















