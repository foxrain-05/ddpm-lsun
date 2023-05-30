import torch
from modules import *
from data import CatDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.T = 1000

        self.betas = torch.linspace(1e-4, 0.02, 1000).to(self.device)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0).to(self.device)


        self.time_embedding = TimeEmbedding(1000, 64, 256)
        self.head = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        self.res1 = ResBlock(64, 64, tdim=256, attention=False)
        self.down1 = DownSample(64)

        self.res2 = ResBlock(64, 128, tdim=256, attention=True)
        self.down2 = DownSample(128)

        self.res3 = ResBlock(128, 256, tdim=256, attention=True)
        self.down3 = DownSample(256)

        self.res4 = ResBlock(256, 512, tdim=256, attention=True)
        self.down4 = DownSample(512)

        self.res5 = ResBlock(512, 1024, tdim=256, attention=True)
        self.down5 = DownSample(1024)

        self.res6 = ResBlock(1024, 1024, tdim=256, attention=True)
        self.res7 = ResBlock(1024, 1024, tdim=256, attention=False)

        self.res8 = ResBlock(2048, 1024, tdim=256, attention=True)
        self.res9 = ResBlock(1024, 512, tdim=256, attention=True)
        self.up1 = Upsample(512)

        self.res10 = ResBlock(1024, 512, tdim=256, attention=True)
        self.res11 = ResBlock(512, 256, tdim=256, attention=True)
        self.up2 = Upsample(256)

        self.res12 = ResBlock(512, 256, tdim=256, attention=True)
        self.res13 = ResBlock(256, 128, tdim=256, attention=True)
        self.up3 = Upsample(128)

        self.res14 = ResBlock(256, 128, tdim=256, attention=True)
        self.res15 = ResBlock(128, 64, tdim=256, attention=True)
        self.up4 = Upsample(64)

        self.res16 = ResBlock(128, 64, tdim=256, attention=False)
        self.res17 = ResBlock(64, 64, tdim=256, attention=False)
        self.up5 = Upsample(64)

        self.tail = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        t = self.time_embedding(t)

        x = self.head(x)

        x = self.res1(x, t)
        x1 = self.down1(x, t)

        x = self.res2(x1, t)
        x2 = self.down2(x, t)

        x = self.res3(x2, t)
        x3 = self.down3(x, t)

        x = self.res4(x3, t)
        x4 = self.down4(x, t)

        x = self.res5(x4, t)
        x5 = self.down5(x, t)
        
        x = self.res6(x5, t)
        x = self.res7(x, t)

        x = torch.cat([x, x5], dim=1)
        x = self.res8(x, t)
        x = self.res9(x, t)
        x = self.up1(x, t)

        x = torch.cat([x, x4], dim=1)
        x = self.res10(x, t)
        x = self.res11(x, t)
        x = self.up2(x, t)
        
        x = torch.cat([x, x3], dim=1)
        x = self.res12(x, t)
        x = self.res13(x, t)
        x = self.up3(x, t)

        x = torch.cat([x, x2], dim=1)
        x = self.res14(x, t)
        x = self.res15(x, t)
        x = self.up4(x, t)

        x = torch.cat([x, x1], dim=1)
        x = self.res16(x, t)
        x = self.res17(x, t)
        x = self.up5(x, t)

        x = self.tail(x)

        return x
    
    def loss(self, x_0):
        x_0 = x_0.to(self.device)
        ts = torch.randint(0, self.T, (x_0.shape[0],), device=self.device)
        eposilon = torch.randn_like(x_0).to(self.device)

        alpha_t_bar = torch.gather(self.alphas_bar, 0, ts)
        x_t = torch.sqrt(alpha_t_bar)[:, None, None, None] * x_0 + torch.sqrt(1 - alpha_t_bar)[:, None, None, None] * eposilon
        
        e_hat = self.forward(x_t, ts)
        loss = F.mse_loss(e_hat.reshape(e_hat.shape[0], -1), eposilon.reshape(eposilon.shape[0], -1), reduction='none')

        return loss
    
    def sample(self, x, t):
        z = torch.randn_like(x) if t > 1 else 0

        e_hat = self.forward(x, t)
        pre_scale = 1 / torch.sqrt(self.alphas[t])
        e_scale = (1 - self.alphas[t]) / torch.sqrt(1 - self.alphas_bar[t])
        post_sigma = torch.sqrt(self.betas[t]) * z
        x = pre_scale * (x - e_scale * e_hat) + post_sigma

        return x



if __name__ == "__main__":

    # batch_size = 4

    # dataset = CatDataset("data/cat")
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = UNet().to(device)
    
    # for i, images in enumerate(dataloader):
    #     images = images.to(device)
    #     t = torch.randint(1000, (batch_size, )).to(device)
    #     x = model(images, t)

    #     save_image(x, "test.png")
    #     break

    model = UNet().to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    t = torch.randint(1000, (1, )).to(device)

    x = model.loss(x)
    print(x.shape)
    save_image(x, "test.png")