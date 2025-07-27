
import os
import sys
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms

# Add path to access encoder and decoder
sys.path.append("F:/ML PROJECT LAB NEW/Image_Transfer_scripts_2")
from server_encoder2 import Encoder
from client_decoder2 import Decoder


# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)
decoder = Decoder().to(device)

encoder.load_state_dict(torch.load("F:/ML PROJECT LAB NEW/Image_Transfer_scripts_2/encoder_custom2.pth", map_location=device))
decoder.load_state_dict(torch.load("F:/ML PROJECT LAB NEW/Image_Transfer_scripts_2/decoder_custom2.pth", map_location=device))
encoder.eval()
decoder.eval()

# Load test image
image_path = "F:/ML PROJECT LAB NEW/received_image.png"  # Replace with any test image
img = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
input_tensor = transform(img).unsqueeze(0).to(device)

# Run through model
with torch.no_grad():
    encoded = encoder(input_tensor)
    decoded = decoder(encoded)

# Convert to numpy
original_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
reconstructed_np = decoded.squeeze().permute(1, 2, 0).cpu().numpy()

# Compute SSIM
score = ssim(original_np, reconstructed_np, data_range=1.0, channel_axis=2)
print(f"SSIM Score: {score:.4f}")

# Optional: save reconstructed image
from matplotlib import pyplot as plt
plt.imsave("reconstructed_eval_output.png", reconstructed_np)
