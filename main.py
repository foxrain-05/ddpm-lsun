import torch
import torch.nn as nn
from data import CatDataset
from torch.utils.data import DataLoader

dataset = CatDataset("data/cat")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



for i, images in enumerate(dataloader):
    print(images.shape)
    break