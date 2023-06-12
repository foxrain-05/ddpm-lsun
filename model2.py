import torch
import torch.nn as nn
from modules1 import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import CatDataset

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.time_emb = TimeEmb(128 * 4) # 512

        self.channels = [128, 256, 512, 1024, 2048]
        self.is_attention = [False, True, True, True]

        self.down = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.down.append(Block(in_channels=self.channels[i], out_channels=self.channels[i + 1], time_channels=512, attention=self.is_attention[i]))
            self.down.append(Block(in_channels=self.channels[i + 1], out_channels=self.channels[i + 1], time_channels=512, attention=self.is_attention[i]))
            self.down.append(DownSample(self.channels[i + 1]))

        self.middle = MiddleBlock(self.channels[-1], time_channels=512)

        self.up = nn.ModuleList()

        self.up.append(UpSample(2048))
        self.up.append(Block(4096, 2048, 512, False))
        self.up.append(Block(2048, 1024, 512, False))
        
    def forward(self, x, t):
        h = self.head(x)
        t = self.time_emb(t)

        hs = [h]
        for down in self.down:
            h = down(h, t)
            hs.append(h)

        h = self.middle(h, t)

        for h in hs:
            print(h.shape)

        print("_" * 40)

        for up in self.up:
            h = up(h, t)
            print(h.shape)
            exit()
            if isinstance(up, Block):
                print(h.shape, hs[-1].shape)
                h = torch.cat([h, hs.pop()], dim=1)
            h = up(h, t)
            print(h.shape)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = DataLoader(CatDataset("data/cat"), batch_size=1, shuffle=True)
    model = UNet().to(device)

    for i, images in enumerate(data_loader):
        images = images.to(device)
        t = torch.arange(0, 1, 1).to(device)
        y = model(images, t)
        break