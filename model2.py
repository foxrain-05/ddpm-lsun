import torch
from modules import *
from data import CatDataset
from torch.utils.data import DataLoader
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding(1000, 128, 512)

        self.head = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.resBlock1 = ResBlock(128, 128, tdim=512)
        self.down1 = DownSample(128)
        self.resBlock2 = ResBlock(128, 256, tdim=512, attention=True)

    def forward(self, x, t):
        x = self.head(x)
        t = self.time_embedding(t)

        x = self.resBlock1(x, t)
        x = self.down1(x, t)

        x = self.resBlock2(x, t)

        return x



if __name__ == "__main__":



    batch_size = 1

    dataset = CatDataset("data/cat")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    
    for i, images in enumerate(dataloader):
        images = images.to(device)
        t = torch.randint(1000, (batch_size, )).to(device)
        x = model(images, t)

        break