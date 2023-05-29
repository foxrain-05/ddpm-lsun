import torch
from modules import *
from data import CatDataset
from torch.utils.data import DataLoader

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding(100, 128, 512)

        self.head = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        x = self.head(x)
        return x



if __name__ == "__main__":
    dataset = CatDataset("data/cat")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    
    for i, images in enumerate(dataloader):
        images = images.to(device)
        x = model(images, 0)
        
        print(x.shape)
        break