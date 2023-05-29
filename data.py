from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from glob import glob


class CatDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = glob(data_path + "/*.jpg")

        self.transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.Resize((512, 512)),

            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        image = Image.open(self.data_path[idx])
        if self.transform:
            image = self.transform(image)
        return image



if __name__ == "__main__":
    data_loader = DataLoader(CatDataset("data/cat"), batch_size=1, shuffle=True)


    for i, images in enumerate(data_loader):
        save_image(images, "{}.png".format(i))
        break

