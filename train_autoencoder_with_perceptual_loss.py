
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# === Dataset Loader ===
class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, f))
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image)

# === Encoder ===
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

# === Decoder ===
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

# === Perceptual Loss Using VGG ===
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:9].eval()  # relu2_2
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return self.criterion(x_vgg, y_vgg)

# === Training Function ===
def train(data_path, epochs=10, batch_size=8, lr=0.001, save_dir="./"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomImageDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    mse_loss = nn.MSELoss()
    perceptual_loss = VGGPerceptualLoss().to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        total_loss = 0

        for batch in dataloader:
            batch = batch.to(device)
            encoded = encoder(batch)
            decoded = decoder(encoded)

            mse = mse_loss(decoded, batch)
            perceptual = perceptual_loss(decoded, batch)
            loss = 0.5 * mse + 0.5 * perceptual  # combine both losses

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(save_dir, "encoder_custom2.pth"))
    torch.save(decoder.state_dict(), os.path.join(save_dir, "decoder_custom2.pth"))
    print("Models saved successfully.")

# === Run Training ===
if __name__ == "__main__":
    train(data_path=r"F:/ML PROJECT/data/natural_images/dog")
