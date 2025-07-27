import socket
import tkinter as tk
from tkinter import filedialog
import time
import torch
import torch.nn as nn
from PIL import Image
import io
import torchvision.transforms as transforms

# ✅ Correct encoder architecture (matches your training notebook)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # [3, 256, 256] → [32, 128, 128]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → [64, 64, 64]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # → [128, 32, 32]
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

# ✅ Load model weights
encoder = Encoder()
encoder.load_state_dict(
    torch.load(r'F:\ML project lab new\Image_Transfer_Scripts\encoder_custom.pth', map_location=torch.device('cpu'))
)
encoder.eval()
print("Encoder model loaded successfully.")

# 📤 Function to encode and send image
def send_image_server(ip, port, image_path):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip, port))
    server_socket.listen(1)
    print(f"Server listening on {ip}:{port}")

    while True:
        data_connection, address = server_socket.accept()
        print(f"Connection from {address}")

        # Load image
        with open(image_path, 'rb') as file:
            image_data = file.read()
            pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        image_tensor = transform(pil_image).unsqueeze(0)  # Shape: [1, 3, 256, 256]

        # Encode image
        encoded_output = encoder(image_tensor)

        # Serialize output
        buffer = io.BytesIO()
        torch.save(encoded_output, buffer)
        encoded_output_bytes = buffer.getvalue()

        # Timestamp for latency tracking
        start_time = time.time()
        message = encoded_output_bytes + b'   ' + str(start_time).encode()

        # Send over socket
        data_connection.sendall(message)
        data_connection.close()
        print("Encoded image sent successfully.")

# 📡 Get server IP
def get_host_ip():
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        return host_ip
    except:
        print("Unable to get Hostname and IP")
        return None

# 🟢 Main execution
if __name__ == "__main__":
    server_ip = get_host_ip()  # Replace with actual IP if needed
    server_port = 55555        # Free port for transmission

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Image to Send")
    if file_path:
        send_image_server(server_ip, server_port, file_path)
    else:
        print("No image selected.")
