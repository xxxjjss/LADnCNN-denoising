import torch
import torch.nn as nn
class UNet1D(nn.Module):
    def __init__(self, input_size=1000):
        super(UNet1D, self).__init__()

        self.enc_conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()


        self.dec_up1 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.dec_up2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)


        self.out_conv = nn.Conv1d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):

        enc1 = self.relu(self.enc_conv1(x.unsqueeze(1)))
        pool1 = self.pool(enc1)
        enc2 = self.relu(self.enc_conv2(pool1))
        pool2 = self.pool(enc2)
        enc3 = self.relu(self.enc_conv3(pool2))

        up1 = self.dec_up1(enc3)
        up1 = torch.cat((up1, enc2), dim=1)
        dec1 = self.relu(self.dec_conv1(up1))

        up2 = self.dec_up2(dec1)
        up2 = torch.cat((up2, enc1), dim=1)
        dec2 = self.relu(self.dec_conv2(up2))


        out = self.out_conv(dec2)
        return out.squeeze(1)