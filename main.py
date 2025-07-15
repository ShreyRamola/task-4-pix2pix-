import os, glob, torch, torchvision
from PIL import Image
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PairedImageDataset(Dataset):
    def __init__(self, folder):
        self.files = glob.glob(f"{folder}/*.jpg")
        self.T = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = self.T(Image.open(self.files[idx]).convert("RGB"))
        return img[:, :, :256], img[:, :, 256:]

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )
    def forward(self, x): return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 4, 2, 1)
        )
    def forward(self, x, y): return self.net(torch.cat([x, y], 1))

def train(model_G, model_D, dataloader, epochs=10):
    G, D = model_G.to(device), model_D.to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4)
    BCE = nn.BCEWithLogitsLoss()
    L1 = nn.L1Loss()

    for epoch in range(epochs):
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            fake = G(x)

            real_pred = D(x, y)
            fake_pred = D(x, fake.detach())

            loss_D = BCE(real_pred, torch.ones_like(real_pred)) + \
                     BCE(fake_pred, torch.zeros_like(fake_pred))

            D.zero_grad(); loss_D.backward(); opt_D.step()

            fake_pred = D(x, fake)
            loss_G = BCE(fake_pred, torch.ones_like(fake_pred)) + L1(fake, y)
            G.zero_grad(); loss_G.backward(); opt_G.step()

        print(f"Epoch {epoch+1}/{epochs} done")

def generate_sample(G, input_img, path="predicted.jpg"):
    G.eval()
    with torch.no_grad():
        img = G(input_img.unsqueeze(0).to(device)).cpu()
        torchvision.utils.save_image(img, path, normalize=True)
        print(f"Saved output to {path}")

dataset = PairedImageDataset("data")
loader = DataLoader(dataset, batch_size=1, shuffle=True)
G = Generator()
D = Discriminator()

train(G, D, loader, epochs=50)        
generate_sample(G, dataset[0][0])     
