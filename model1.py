import torch
from modules import *

class UNet(nn.Module):
    def __init__(self, T, ch, attn, num_blocks, ch_mult, dropout=0.1):
        super().__init__()

        tdim = ch * 4
        self.time_emb = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()

        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult

            for _ in range(num_blocks):
                resblock = ResBlock(now_ch, out_ch, tdim=tdim, dropout=dropout, attention=(i in attn))
                self.downblocks.append(resblock)
                now_ch = out_ch
                chs.append(now_ch)

            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)
        
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim=tdim, dropout=dropout, attention=True),
            ResBlock(now_ch, now_ch, tdim=tdim, dropout=dropout, attention=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in enumerate(ch_mult[::-1][1:]):
            out_ch = ch * mult
            for _ in range(num_blocks + 1):
                resblock = ResBlock(now_ch, out_ch, tdim=tdim, dropout=dropout, attention=(i in attn))
                self.upblocks.append(resblock)
                now_ch = out_ch
