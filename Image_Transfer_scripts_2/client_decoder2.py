import io
import socket
import time
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ✅ Decoder class matching what was used during training
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


# ✅ Load trained weights
decoder = Decoder()
decoder.load_state_dict(
    torch.load(r"F:\ML project lab new\Image_Transfer_scripts_2\decoder_custom2.pth", map_location=torch.device("cpu"))
)
decoder.eval()
print("Decoder loaded successfully.")


def receive_image_client(server_ip, server_port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))

    image_data = b""
    while True:
        chunk = client_socket.recv(1024)
        if not chunk:
            break
        image_data += chunk

    # Parse and decode the received message
    try:
        encoded_image_data, server_start_time = image_data.split(b"   ")
        start_time = float(server_start_time.decode())
    except ValueError:
        print("Data formatting issue. Ensure server is sending 'data + delimiter + start_time'")
        return

    transfer_time = time.time() - start_time
    print(f"Image received successfully in {transfer_time:.3f} seconds")
    client_socket.close()

    received_encoded_output = torch.load(io.BytesIO(encoded_image_data), map_location=torch.device("cpu"))
    display_received_image(received_encoded_output)


def display_received_image(encoded_output):
    decoded_output = decoder(encoded_output)
    decoded_output_np = decoded_output.detach().numpy()
    img = np.transpose(decoded_output_np[0], (1, 2, 0))

    # Save and show
    plt.imsave('received_image.png', img)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.axis("off")
    plt.title("Received Image")
    plt.show()


def get_host_ip():
    try:
        return socket.gethostbyname(socket.gethostname())
    except:
        print("Unable to get Hostname and IP")
        return None


if __name__ == "__main__":
    server_ip = get_host_ip()  # Replace with actual server IP if not local
    server_port = 55555
    receive_image_client(server_ip, server_port)
