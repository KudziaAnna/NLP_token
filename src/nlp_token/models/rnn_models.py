from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..configs import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRUNet(nn.Module):
    def __init__(self, cfg: Config):
        super(GRUNet, self).__init__()
        self.cfg = cfg

        self.input_size = self.cfg.experiment.input_size
        self.output_size = self.cfg.experiment.output_size
        self.hidden_size = self.cfg.experiment.hidden_size
        self.n_layers =self.cfg.experiment.n_layers
        self.drop_prob = self.cfg.experiment.dropout[1]
        
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.n_layers, batch_first=True, dropout=self.drop_prob)
        self.fc = nn.Linear(self.hidden_size, self.output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device)
        return hidden

class LSTMNet(nn.Module):
    def __init__(self, cfg: Config):
        super(LSTMNet, self).__init__()
        self.cfg = cfg

        self.input_size = self.cfg.experiment.input_size
        self.output_size = self.cfg.experiment.output_size
        self.hidden_size = self.cfg.experiment.hidden_size
        self.n_layers =self.cfg.experiment.n_layers
        self.drop_prob = self.cfg.experiment.dropout[1]
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, batch_first=True, dropout=self.drop_prob)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_size.zero_().to(device)))
        return hidden