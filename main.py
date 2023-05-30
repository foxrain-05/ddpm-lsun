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


for epoch in range(200):
    model.train()
    for i, x in enumerate(dataloader):

        x = x.to(device)
        x = torch.randn(4, 3, 256, 256).to(device)
        optimizer.zero_grad()

        loss = model.loss(x).mean()
        loss.backgrad()
        optimizer.step()

        print(f"Epoch: {epoch} | Batch: {i} | Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 256, 256).to(device)
        ts = torch.arange(1000, -1, 0, -1).to(device)
        
        for t in ts:
            x = model.sample(x, t)

        save_image(x, f"test/{epoch}.png")
    
    torch.save(model.state_dict(), f"checkpoints/{epoch}.pt")
    
    