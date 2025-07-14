import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        # x: [B, T_in, F]
        out, _ = self.lstm(x)  # [B, T_in, H]
        out = out[:, -1, :]    # 取最后一个时间步输出
        out = self.fc(out)     # [B, pred_len]
        return out.unsqueeze(-1)  # [B, pred_len, 1]
