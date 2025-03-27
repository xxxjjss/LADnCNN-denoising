from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
snr_levels = [-15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
excel_file = 'MS-signal'
df = pd.read_excel(excel_file, sheet_name='Sheet1')
signals = df.values
target_length = 1000
resampled_signals = np.zeros((signals.shape[0], target_length))
for i in range(signals.shape[0]):
    interp_func = interp1d(np.arange(signals.shape[1]), signals[i], kind='linear')
    resampled_signals[i] = interp_func(np.linspace(0, signals.shape[1] - 1, target_length))
signals = resampled_signals
def add_noise_to_signal(signal, snr_dB):
    signal_power = np.sum(signal ** 2) / len(signal)
    noise_power = signal_power / (10 ** (snr_dB / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise
s = []
s = []
for signal in signals:
    for snr_db in snr_levels:
        clean_signal = signal
        noisy_signal = add_noise_to_signal(signal, snr_db)
        s.append(clean_signal)
        s.append(noisy_signal)
clean_data = np.array(s)
noisy_data = np.array(s)
split_ratio = 0.8
split_index = int(split_ratio * len(clean_data))
clean_train = clean_data[:split_index]
noisy_train = noisy_data[:split_index]
clean_test = clean_data[split_index:]
noisy_test = noisy_data[split_index:]
clean_train_tensor = torch.tensor(clean_train).float()
noisy_train_tensor = torch.tensor(noisy_train).float()
clean_test_tensor = torch.tensor(clean_test).float()
noisy_test_tensor = torch.tensor(noisy_test).float()
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
batch_size = 32
train_dataset = TensorDataset(noisy_train_tensor, clean_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DnCNN1D(input_size = 1000).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 12
num_epochs = 70
train_loss_history = []
test_loss_history = []
noisy_test_tensor = noisy_test_tensor.to(device)
clean_test_tensor = clean_test_tensor.to(device)
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * inputs.size(0)
    epoch_train_loss = running_train_loss / len(train_dataset)
    train_loss_history.append(epoch_train_loss)
    model.eval()
    with torch.no_grad():
        denoised_output = []
        for i in range(0, len(noisy_test_tensor), batch_size):
            output = model(noisy_test_tensor[i:i + batch_size])
            denoised_output.append(output)
            torch.cuda.empty_cache()
        denoised_output = torch.cat(denoised_output, dim=0)
        test_loss = criterion(denoised_output, clean_test_tensor.to(device)).item()
        test_loss_history.append(test_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.6f}, Test Loss: {test_loss:.6f}")
plt.figure(figsize=(10, 6))
plt.plot(train_loss_history, label='Train Loss', color='blue')
plt.plot(test_loss_history, label='Test Loss', color='red')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Testing Loss', fontsize=14)
plt.legend(fontsize=14)
plt.show()
torch.save(model.state_dict(), 'LAD-DnCNN.pth')

