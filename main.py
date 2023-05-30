import torch
from data import CatDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import UNet

dataset = CatDataset("data/cat")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


for epoch in range(100):
    model.train()
    for i, x in enumerate(dataloader):
        x = x.to(device)
        optimizer.zero_grad()

        loss = model.loss(x)
        loss.backgrad()
        optimizer.step()

        print(f"Epoch: {epoch} | Batch: {i} | Loss: {loss.item()}")

    model.eval()
    