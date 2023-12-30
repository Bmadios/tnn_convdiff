import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers,
                                          dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.fc_in = nn.Linear(input_dim, d_model)
        self.fc_out = nn.Linear(d_model, 1)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

    def forward(self, src):
        src = self.fc_in(src)  # Linear transformation
        src = self.pos_encoder(src)  # Positional Encoding
        output = self.transformer(src, src)  # Transformer
        output = self.fc_out(output)  # Output linear layer
        output = output[:, -1, :] 
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
