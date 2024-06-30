import torch
from torch.nn import Module, Sequential, Conv1d, BatchNorm1d, ReLU, MaxPool1d, Linear, Dropout1d, LSTM, Flatten
import torch.nn.functional as F

class AudioCNN(Module):
    def __init__(self): # in size: 162
        super(AudioCNN, self).__init__()
        self.conv1 = Sequential(
            Conv1d(1, 64, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(64),
            ReLU(),
            Conv1d(64, 64, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(64),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2)
        ) # out size: 81
        self.conv2 = Sequential(
            Conv1d(64, 128, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(128),
            ReLU(),
            Conv1d(128, 128, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(128),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2)
        ) # out size: 40
        self.conv3 = Sequential(
            Conv1d(128, 256, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(256),
            ReLU(),
            Conv1d(256, 256, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(256),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2)
        ) # out size:  20
        self.conv4 = Sequential(
            Conv1d(256, 512, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(512),
            ReLU(),
            Conv1d(512, 512, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(512),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2)
        ) # out size: 10

        self.fc1 = Sequential(
            Linear(512*10, 1024),
            ReLU(),
            Dropout1d(0.2),
            Linear(1024, 512),
            ReLU(),
            Dropout1d(0.2),
            Linear(512, 8),
            # ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class MediumCNN(Module):
    def __init__(self):
        super(MediumCNN, self).__init__()
        self.conv1 = Sequential( # 162
            Conv1d(1, 8, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(8),
            ReLU(),
            Conv1d(8, 8, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(8),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2)
        ) # 81

        self.conv2 = Sequential( 
            Conv1d(8, 16, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(16),
            ReLU(),
            Conv1d(16, 16, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(16),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2)
        ) # 40

        self.conv3 = Sequential(
            Conv1d(16, 32, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(32),
            ReLU(),
            Conv1d(32, 32, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(32),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2)
        ) # 20

        self.fc1 = Sequential(
            Linear(16*40, 64),
            ReLU(),
            Dropout1d(0.2),
            Linear(64, 8),
            # ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
class SimpleAudioCNN(Module):
    def __init__(self):
        super(SimpleAudioCNN, self).__init__()
        self.conv1 = Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool = MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = Linear(128*20, 512)  # Adjust 81 based on your input size
        self.fc2 = Linear(512, 8)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        return x

class SimpleAudioCNN2(Module):
    def __init__(self):
        super(SimpleAudioCNN2, self).__init__()
        self.conv1 = Sequential(
            Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc1 = Sequential(
            Linear(16*81, 8),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class AudioLSTM(Module):
    def __init__(self, input_size, hidden_size=256, num_layers=4):
        super(AudioLSTM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = Linear(hidden_size, 8)
        self.relu = ReLU()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        # out = self.relu(out)

        return out