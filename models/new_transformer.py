import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TemporalAttention(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = x.transpose(1, 2)  # [B, d_model, T]
        x = self.conv(x)       # [B, d_model, T]
        x = x.transpose(1, 2)  # [B, T, d_model]
        x = self.norm(x)
        x = self.activation(x)
        return x

class EnhancedTransformerForecast(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, pred_len=90, dropout=0.1, model_variant="original"):
        super().__init__()
        self.model_variant = model_variant
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        
        if model_variant == "original":
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:  # enhanced
            self.temporal_attn = TemporalAttention(d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, pred_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        
        if self.model_variant == "enhanced":
            x = self.temporal_attn(x)
        
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])  # Take last time step
        x = self.dropout(x)
        x = self.fc(x).unsqueeze(-1)  # [B, pred_len, 1]
        return x