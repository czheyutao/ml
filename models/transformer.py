import torch
import torch.nn as nn

class TransformerForecast(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, pred_len=90):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 500, d_model))  # 假设最长长度 < 500
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_enc[:, :x.size(1)]  # [B, T_in, d_model]
        x = self.encoder(x)  # [B, T_in, d_model]
        out = x[:, -1, :]    # 最后一个时间步
        out = self.fc(out).unsqueeze(-1)  # [B, pred_len, 1]
        return out
