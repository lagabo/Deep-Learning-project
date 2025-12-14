import torch
import torch.nn as nn
import config

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_classes=6):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        logits = self.fc(last_output)
        return logits

class CNN1DClassifier(nn.Module):
    def __init__(self, num_classes=6, kernel_size=3, channels=16):
        super(CNN1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels*2, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels*2)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(channels*2, num_classes)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.gap(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

def get_lstm_model():
    return LSTMClassifier(
        input_size=1,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_classes=config.NUM_CLASSES
    )

def get_cnn_model():
    return CNN1DClassifier(
        num_classes=config.NUM_CLASSES,
        kernel_size=config.CNN_KERNEL_SIZE,
        channels=config.CNN_CHANNELS
    )
