import torch.nn as nn
import torch.nn.functional as F
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv1d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=1)
    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn
class LCA(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.proj_2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.proj_3 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model, kernel_size=1)
        )
    def forward(self, x):
        x1 = F.leaky_relu(self.proj_1(x))
        y = self.proj_3(x1)
        y = x1 * y
        y = self.proj_2(y)
        return x + y
class LLA(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = x + shortcut
        x = self.proj_2(x)
        return x
class DnCNN1D(nn.Module):
    def __init__(self, input_size, num_of_layers=17):
        super(DnCNN1D, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(
            nn.Conv1d(in_channels=1, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        layers.append(LCA(features))

        layers.append(
            nn.Conv1d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      bias=False))
        layers.append(nn.BatchNorm1d(features))
        layers.append(nn.ReLU(inplace=True))

        layers.append(LCA(features))

        for _ in range(17):
            layers.append(
                nn.Conv1d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm1d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(LLA(features))

        layers.append(
            nn.Conv1d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      bias=False))
        layers.append(nn.ReLU(inplace=True))

        layers.append(LLA(features))

        layers.append(
            nn.Conv1d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x.unsqueeze(1))
        return out.squeeze(1)