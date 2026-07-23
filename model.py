import cv2
import torch
import torch.nn as nn
import numpy as np


class UltraTightAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        self.encoder_linear = nn.Linear(
            128 * 14 * 14,
            latent_dim
        )

        self.decoder_linear = nn.Linear(
            latent_dim,
            128 * 14 * 14
        )

        self.relu = nn.ReLU()

        self.unflatten = nn.Unflatten(
            1,
            (128, 14, 14)
        )

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                16,
                3,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_linear(x)

        x = self.decoder_linear(x)
        x = self.relu(x)
        x = self.unflatten(x)

        x = self.decoder_cnn(x)

        return x


def load_and_preprocess_image(
    image_path,
    image_size,
    device
):
    img = cv2.imread(image_path)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img_resized = cv2.resize(
        img,
        (image_size, image_size)
    )

    img_normalized = (
        img_resized.astype(np.float32) / 255.0
    )

    tensor_input = torch.tensor(
        np.transpose(img_normalized, (2, 0, 1))
    ).unsqueeze(0).to(device)

    return tensor_input


def run_autoencoder(model, tensor_input):
    with torch.no_grad():
        tensor_output = model(tensor_input)

        tensor_residual = torch.relu(
            tensor_input - tensor_output
        )

    return tensor_output, tensor_residual