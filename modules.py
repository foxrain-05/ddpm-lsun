import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, t):
        self.emb = torch.arange(0, self.d_model, 2, device=self.device).float() / self.d_model * math.log(10000)
        self.emb = torch.exp(-self.emb)

        pos_enc = t.repeat(1, self.d_model // 2).to(self.device) * self.emb
        pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)
        pos_enc = pos_enc[:, :, None, None]

        return pos_enc
    

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        super().__init__()
        emb = torch.arange(0, d_model, 2).float() / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)

        self.time_emb = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        return self.time_emb(t)
    

class DownSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, temb):
        return self.conv(x)
    

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, temb): 
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
    

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

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, tdim, dropout=0.1, attention=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.temb = nn.Sequential(
            nn.GELU(),
            nn.Linear(tdim, out_channels),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

        if attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, temb):
        h = self.block1(x)
        h = h + self.temb(temb)[:, :, None, None]
        h = self.block2(h)

        h =  h + self.shortcut(x)
        h = self.attention(h)

        return h
    


if __name__ == '__main__':
    # tast ResBlock
    x = torch.randn(1, 32, 64, 64)
    temb = torch.randn(1, 256)
    block = ResBlock(32, 32, 256)
    y = block(x, temb)
    
    