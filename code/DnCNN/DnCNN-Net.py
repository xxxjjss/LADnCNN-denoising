import torch.nn as nn
class DnCNN1D(nn.Module):
    def __init__(self, input_size, num_of_layers=17):
        super(DnCNN1D, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv1d(in_channels=1, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv1d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm1d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv1d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x.unsqueeze(1))  # Unsqueeze to add a channel dimension
        return out.squeeze(1)  # Squeeze the channel dimension to get 1D output
