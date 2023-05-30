import torch
from data import CatDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import UNet

dataset = CatDataset("data/cat")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


for epoch in range(200):
    model.train()
    for i, x in enumerate(dataloader):

        x = x.to(device)
        optimizer.zero_grad()

        loss = model.loss(x).mean()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch} | Batch: {i} | Loss: {loss.item()}")


        if i % 1000 == 0:
            model.eval()
            with torch.no_grad():
                x = torch.randn(1, 3, 256, 256, device=device)
                ts = torch.arange(999, 0, -1, device=device)[:, None]
                
                for t in ts:
                    x = model.sample(x, t)

                save_image(x, f"test/{epoch}_{i}.png")
    
    torch.save(model.state_dict(), f"checkpoints/{epoch}.pt")
    
    