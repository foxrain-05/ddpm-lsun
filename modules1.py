import torch
from torch import nn
from torch.nn import functional as F
import math


class TimeEmb(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

        self.linear1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t):
        half_dim = self.n_channels // 8
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(-emb * torch.arange(half_dim, device=t.device).float())
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        emb = self.linear1(emb)
        emb = self.act(emb)
        emb = self.linear2(emb)

        return emb


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gn = nn.GroupNorm(32, in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.gn(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
    
        q = q.permute(0, 2, 3, 1).view(B, H*W, C)
        k = k.view(B, C, H * W)

        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H*W, C)
        h = torch.bmm(w, v)
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.out(h)
        
        return x + h


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb, n_groups=32, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb = temb
        self.n_groups = n_groups
        
        self.gn1 = nn.GroupNorm(self.n_groups, self.in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)

        self.gn2 = nn.GroupNorm(self.n_groups, self.out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.temb = nn.Linear(self.temb, self.out_channels)
        self.temb_act = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        h = self.gn1(x)
        h = self.act1(h)
        h = self.conv1(h)

        h += self.temb(self.temb_act(t))[:, :, None, None]

        h = self.gn2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, attention=False):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = nn.Identity()
        
    def forward(self, x, t):
        h = self.res(x, t)
        h = self.attention(h)
        return h
    

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, attention=False):
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attention(x)
        return x
    

class MiddleBlock(nn.Module):
    def __init__(self, n_channels, time_channels):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attention = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.attention(x)
        x = self.res2(x, t)
        return x
    
    
class UpSample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels * 2, n_channels * 2, kernel_size=3, padding=1)

    def forward(self, x, t=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class DownSample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t=None):
        x = self.conv(x)
        return x

if __name__ == "__main__":
    # test layers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test time embedding
    temb = TimeEmb(256).to(device)
    t = torch.arange(0, 1000, 1).to(device)
    print(temb(t).shape)

    # test residual block
    res = ResidualBlock(256, 256, 256).to(device)
    x = torch.randn(1, 256, 32, 32).to(device)
    t = torch.randn(1, 256).to(device)
    print(res(x, t).shape)

    # test down block
    down = DownBlock(256, 512, 256).to(device)
    x = torch.randn(1, 256, 32, 32).to(device)
    t = torch.randn(1, 256).to(device)
    print(down(x, t).shape)

    # test up block
    up = UpBlock(256, 512, 256).to(device)
    x = torch.randn(1, 512 + 256, 32, 32).to(device)
    t = torch.randn(1, 256).to(device)
    print(up(x, t).shape)

    # test middle block
    middle = MiddleBlock(256, 256).to(device)
    x = torch.randn(1, 256, 32, 32).to(device)
    t = torch.randn(1, 256).to(device)
    print(middle(x, t).shape)

    # test upsample
    upsample = UpSample(256).to(device)
    x = torch.randn(1, 256, 32, 32).to(device)
    print(upsample(x).shape)

    # test downsample
    downsample = DownSample(256).to(device)
    x = torch.randn(1, 256, 32, 32).to(device)
    print(downsample(x).shape)

    # test attention
    attention = AttentionBlock(256).to(device)  
    x = torch.randn(1, 256, 32, 32).to(device)
    print(attention(x).shape)



