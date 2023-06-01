import torch
import torch.nn as nn

from modules1 import *

class UNet(nn.Module):
    def __init__(self, n_channels=64):
        super().__init__()

        self.head = nn.Conv2d(3, n_channels, kernel_size=3, padding=1)
        self.time_emb = TimeEmb(n_channels * 4)

        self.is_attentions = [False, False, True, True, True]
        self.mults = [1, 2, 2, 4, 8]

        in_channels  = n_channels
        self.down = nn.ModuleList()
        for i in range(len(self.mults)):
            out_channels = n_channels * self.mults[i]

            for _ in range(2):
                layer = DownBlock(in_channels=in_channels, out_channels=out_channels, time_channels=n_channels * 4, attention=self.is_attentions[i])
                self.down.append(layer)
                in_channels = out_channels

            if i < len(self.mults) - 1:
                self.down.append(DownSample(in_channels))


        self.middle = MiddleBlock(out_channels, time_channels=n_channels * 4)


    def forward(self, x, t):
        h = self.head(x)

        print(t.shape)
        t = self.time_emb(t)
        print(t.shape)
        return h
    

if __name__ == "__main__":
    model = UNet()
    x = torch.randn(1, 3, 256, 256)
    t = torch.arange(1, 10, 1)
    model(x, t)
