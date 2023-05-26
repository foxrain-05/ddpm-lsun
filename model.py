import torch
from modules import *


class DiffusionModel(nn.Module):
    def __init__(self, t_range=1000, beta_small=1e-4, beta_large=0.02):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t_range = t_range
        self.beta_small = beta_small
        self.beta_large = beta_large

        self.betas = torch.linspace(beta_small, beta_large, t_range).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0, dtype=torch.float32).to(self.device)

        self.inc = DoubleConv(3, 64)
        self.outc = OutConv(64, 3)

        self.donw1 = Down(64, 128)
        self.donw2 = Down(128, 256)
        self.donw3 = Down(256, 512)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)

        self.sa1 = SAWrapper(256, 8)
        self.sa2 = SAWrapper(512, 4)
        self.sa3 = SAWrapper(256, 8)

        self.pe1 = PositionalEncoding(128)
        self.pe2 = PositionalEncoding(256)
        self.pe3 = PositionalEncoding(512)
        self.pe4 = PositionalEncoding(256)
        self.pe5 = PositionalEncoding(128)
        self.pe6 = PositionalEncoding(64)
    
    def forward(self, x, t):
        x1 = self.inc(x) # x1 shape [batch, 64, 32, 32]
        x2 = self.donw1(x1) + self.pe1(t) # x2 shape [batch, 128, 16, 16]
        x3 = self.donw2(x2) + self.pe2(t) # x3 shape [batch, 256, 8, 8]
        x3 = self.sa1(x3)
        x4 = self.donw3(x3) + self.pe3(t) # x4 shape [batch, 512, 4, 4]
        x4 = self.sa2(x4)
        
        x = self.up1(x4) + x3 + self.pe4(t) # x shape [batch, 256, 8, 8]
        x = self.sa3(x) 
        x = self.up2(x) + x2 + self.pe5(t) # x shape [batch, 128, 16, 16]
        x = self.up3(x) + x1 + self.pe6(t) # x shape [batch, 64, 32, 32]

        x = self.outc(x)
        return x

    def loss_fn(self, x_0):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """

        x_0 = x_0.to(self.device)
        ts = torch.randint(0, self.t_range, size=(x_0.shape[0],), dtype=torch.int64, device=self.device)
        eposilon = torch.randn_like(x_0, dtype=torch.float32, device=self.device)
        
        alpha_t_bar = torch.gather(self.alphas_bar, 0, ts)
        x_t = torch.sqrt(alpha_t_bar)[:, None, None, None] * x_0 + torch.sqrt(1 - alpha_t_bar)[:, None, None, None] * eposilon

        e_hat = self.forward(x_t, ts[:, None].type(torch.float))

        loss = F.mse_loss(e_hat.reshape(e_hat.shape[0], -1), eposilon.reshape(eposilon.shape[0], -1))

        return loss
    
    def sample(self, x, t):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        z = torch.randn_like(x, dtype=torch.float32) if t > 1 else 0
        
        e_hat = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1))
        pre_scale = 1 / torch.sqrt(self.alphas[t])
        e_scale = (1 - self.alphas[t]) / torch.sqrt(1 - self.alphas_bar[t])
        post_sigma = torch.sqrt(self.betas[t]) * z
        x = pre_scale * (x - e_scale * e_hat) + post_sigma

        return x
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionModel().to(device)
    x = torch.randn(1, 3, 32, 32).to(device)
    print(model.loss_fn(x))