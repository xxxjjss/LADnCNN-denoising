import torch.nn as nn
class Autoencoder1D(nn.Module):
    def __init__(self):
        super(Autoencoder1D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=4, stride=2, padding=1),
            nn.Conv1d(12, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(24, 12, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(12, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x