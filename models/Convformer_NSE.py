import torch
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class Dconv(nn.Module):
    def __init__(self, kernel_size, stride, in_channels, out_channels, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.act1 = h_swish()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn(x)
        return x


class conv_projection(nn.Module):
    def __init__(self, in_channels, conv_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.conv_ratio = conv_ratio
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=in_channels * conv_ratio,
                      kernel_size=1, stride=1, padding=0),
            h_swish())
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels * conv_ratio,
                      out_channels=in_channels,
                      kernel_size=1, stride=1, padding=0),
            h_swish())
        self.norm = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return x + self.norm(out)


class Sparse_Attention_and_conv_mlp(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 kv_ratio=1,
                 conv_ratio=4,
                 proj_drop_ratio=0.,
                 attn_drop_ratio=0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.heads_dim = dim // num_heads
        self.scale = qk_scale or self.heads_dim ** -0.5
        self.q = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(self.dim, 2 * self.dim, bias=qkv_bias)
        self.resize = nn.Linear(self.dim, self.dim)
        self.mlp = conv_projection(in_channels=dim, conv_ratio=conv_ratio)
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop_ratio, inplace=True)
        if kv_ratio > 1:
            self.reduce = nn.Sequential(nn.Conv1d(in_channels=2 * dim, out_channels=2 * dim,
                                                  kernel_size=kv_ratio, stride=kv_ratio),
                                        h_swish(),
                                        nn.BatchNorm1d(2 * dim))
        else:
            self.reduce = nn.Identity()

    def forward(self, x):
        b, n, c = x.shape
        # [b, n, c] ==> [b, num_heads, n, heads_dim]
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        # [b, num_heads, n, heads_dim]
        # [b, n, 2*c]
        kv = self.kv(x)
        # [b, 2 * c, n]
        kv = kv.transpose(-1, -2)
        # [b, 2 * c, m] ==> [b, m, 2*c] ==> [b, m, 2, num_heads, heads_dim] ==> [2, b, num_heads, m, heads_dim]
        kv = self.reduce(kv).transpose(-1, -2).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3,
                                                                                                              1, 4)
        # [b, num_heads, m, heads_dim]
        k, v = kv[0], kv[1]
        # [b, num_heads, n, m]
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn - self.attn_drop(attn)
        # [b, num_heads, n, heads_dim] ==> [b, n, num_heads, heads_dim] ==> [b, n, dim]
        value = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        value = self.resize(value)
        x_mlp = (x + value)
        x_mlp = self.norm1(x_mlp)
        x_out = self.mlp(x_mlp.transpose(-1, -2)).transpose(-1, -2)
        out = self.norm2(x_mlp + x_out)
        return out.transpose(-1, -2)


class Convformer_Block(nn.Module):
    def __init__(self, dim,
                 stride, kernel_size, in_channels,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 kv_ratio=2,
                 conv_ratio=4,
                 proj_drop_ratio=0.,
                 attn_drop_ratio=0):
        super().__init__()
        assert dim == in_channels
        self.conv1 = Dconv(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                           stride=stride, padding=0)
        self.conv2 = Dconv(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3,
                           stride=1, padding=1)
        self.conv3 = Dconv(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                           padding=0)
        self.attention_and_mlp = Sparse_Attention_and_conv_mlp(dim=dim,
                                                               num_heads=num_heads,
                                                               qkv_bias=qkv_bias,
                                                               qk_scale=qk_scale,
                                                               kv_ratio=kv_ratio,
                                                               conv_ratio=conv_ratio,
                                                               proj_drop_ratio=proj_drop_ratio,
                                                               attn_drop_ratio=attn_drop_ratio)

    def forward(self, x):
        f = self.conv1(x)
        f = self.conv2(f)
        f = self.conv3(f)
        f = f.transpose(-1, -2)
        out = self.attention_and_mlp(f)
        return out


class Avg_max_channel_attention(nn.Module):
    def __init__(self, in_channel, ratio):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp_avg = nn.Sequential(nn.Linear(in_features=in_channel,
                                               out_features=in_channel * ratio),
                                     nn.ReLU(True),
                                     nn.Linear(in_features=in_channel * ratio,
                                               out_features=in_channel),
                                     nn.Softmax(dim=-1))
        self.mlp_max = nn.Sequential(nn.Linear(in_features=in_channel,
                                               out_features=in_channel * ratio),
                                     nn.ReLU(True),
                                     nn.Linear(in_features=in_channel * ratio,
                                               out_features=in_channel),
                                     nn.Softmax(dim=-1))

    def forward(self, x):
        # [b, c, n] ==> [b, c, 1]
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        # [b, c, 1] ==> [b, 1, c]
        x_avg = x_avg.transpose(-2, -1)
        x_max = x_max.transpose(-2, -1)
        x_avg_attn = self.mlp_avg(x_avg)
        x_max_attn = self.mlp_max(x_max)
        x_avg_attn = x_avg_attn.transpose(-2, -1)
        x_max_attn = x_max_attn.transpose(-2, -1)
        return x_avg_attn * x + x_max_attn * x


class ConvformerStage(nn.Module):
    def __init__(self,
                 depth,
                 dim,
                 kernel_size, stride, in_channels,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 kv_ratio=2,
                 conv_ratio=4,
                 proj_drop_ratio=0.,
                 attn_drop_ratio=0
                 ):
        super().__init__()
        self.blocks = nn.Sequential(*[
            Convformer_Block(dim=dim,
                             in_channels=in_channels,
                             kernel_size=kernel_size,
                             stride=stride if i == 0 else 1,
                             num_heads=num_heads,
                             qkv_bias=qkv_bias,
                             qk_scale=qk_scale,
                             kv_ratio=kv_ratio,
                             conv_ratio=conv_ratio,
                             proj_drop_ratio=proj_drop_ratio,
                             attn_drop_ratio=attn_drop_ratio) for i in range(depth)])

    def forward(self, x):
        return self.blocks(x)


class Convformer_nse(nn.Module):
    def __init__(self,
                 in_c, num_cls,
                 h_args, cfg,
                 use_init=False,
                 use_last=True):
        super().__init__()
        self.use_last = use_last
        self.stem = Dconv(in_channels=in_c, out_channels=cfg[0][3], kernel_size=3, stride=1, padding=0)
        self.stage1 = self._make_layer(cfg[0])
        self.stage2 = self._make_layer(cfg[1])
        self.stage3 = self._make_layer(cfg[2])

        self.patch_merging1 = nn.Sequential(
            nn.Conv1d(in_channels=cfg[0][3],
                      out_channels=cfg[1][3],
                      kernel_size=1),
            nn.BatchNorm1d(cfg[1][3]))
        self.patch_merging2 = nn.Sequential(
            nn.Conv1d(in_channels=cfg[1][3],
                      out_channels=cfg[2][3],
                      kernel_size=1),
            nn.BatchNorm1d(cfg[2][3]))
        if use_last:
            self.last_channels = cfg[2][3]
        else:
            self.last_channels = cfg[0][3] + cfg[1][3] + cfg[2][3]
        self.nse = Avg_max_channel_attention(in_channel=self.last_channels, ratio=4)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.ModuleList()
        if not h_args:
            self.classifier.append(nn.Linear(self.last_channels, num_cls))
            self.classifier.append(nn.Softmax(dim=-1))
        else:
            for i in range(len(h_args)):
                if i == 0:
                    self.classifier.append(nn.Linear(self.last_channels, h_args[i]))
                    self.classifier.append(h_swish())
                    # self.classifier.append(nn.Dropout(p=0))
                else:
                    self.classifier.append(nn.Linear(h_args[i - 1], h_args[i]))
                    self.classifier.append(h_swish())
                    # self.classifier.append(nn.Dropout(p=0))
            self.classifier.append(nn.Linear(h_args[-1], num_cls))
            # self.classifier.append(nn.Softmax(dim=-1))
        if use_init:
            self._initialize_weights()

    def _make_layer(self, cfg):
        layers = []
        layers += [ConvformerStage(depth=cfg[0],
                                    kernel_size=cfg[1],
                                    stride=cfg[2],
                                    in_channels=cfg[3],
                                    dim=cfg[3],
                                    num_heads=cfg[4])]
        return nn.Sequential(*layers)

    def forward(self, x):
        b = x.shape[0]
        x = self.stem(x)
        x = self.stage1(x)
        f0 = x
        x = self.patch_merging1(x)
        x = self.stage2(x)
        f1 = x
        x = self.patch_merging2(x)
        x = self.stage3(x)
        f2 = x
        if self.use_last:
            x = self.nse(x)
        else:
            f0 = F.interpolate(f0, size=f2.size()[-1])
            f1 = F.interpolate(f1, size=f2.size()[-1])
            x = torch.cat([f0, f1, f2], dim=1)
            x = self.nse(x)
        # [b, c1, n] ==> [b, c1, 1]
        x = self.avg_pool(x)
        x = x.view(b, -1)
        for module in self.classifier:
            x = module(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


cfgs = {"S": [[1, 2, 5, 16, 2], [1, 2, 3, 32, 2], [1, 1, 3, 64, 4]],
        "M": [],
        "B": []}


def convoformer_v1_small(_, in_channel, out_channel):
    model = Convformer_nse(cfg=cfgs["S"],
                           h_args=[100, 64, 32],
                           in_c=in_channel,
                           num_cls=out_channel)
    return model


def convoformer_v1_middle(in_c, h_args, num_cls):
    model = Convformer_nse(cfg=cfgs["M"],
                           h_args=h_args,
                           in_c=in_c,
                           num_cls=num_cls)
    return model


def convormer_v1_big(in_c, h_args, num_cls):
    model = Convformer_nse(cfg=cfgs["B"],
                           h_args=h_args,
                           in_c=in_c,
                           num_cls=num_cls)
    return model
