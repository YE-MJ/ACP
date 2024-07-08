import torch
import torch.nn.functional as F
import torch.nn as nn

class SequenceModel_deep(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(SequenceModel_deep, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim*2)
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.bn1(lstm_out)
        x = F.relu(self.fc1(lstm_out))
        x = self.bn2(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output