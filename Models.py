import torch
from torch.nn import Module, Sequential, Conv1d, BatchNorm1d, ReLU, MaxPool1d, Linear, Dropout1d, LSTM, Flatten, Dropout, GELU
import torch.nn.functional as F
import TrainTesting

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
        self.batchnorm1 = BatchNorm1d(32)
        self.conv3 = Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = BatchNorm1d(64)
        self.conv5 = Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = BatchNorm1d(128)
        self.pool = MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = Linear(128*20, 1024)  # Adjust 81 based on your input size
        self.fc2 = Linear(1024, 512)
        self.fc3 = Linear(512, 8)
        self.dropout = Dropout1d(0.2)
    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool(self.batchnorm1(F.relu(self.conv2(x))))
        x = F.relu(self.batchnorm2(self.conv3(x)))
        x = self.pool(self.batchnorm2(F.relu(self.conv4(x))))
        x = F.relu(self.batchnorm3(self.conv5(x)))
        x = self.pool(self.batchnorm3(F.relu(self.conv6(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
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
    
class CNNModel(Module):
    def __init__(self, input_shape):
        super(CNNModel, self).__init__()
        
        self.conv1 = Conv1d(in_channels=1, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.pool1 = MaxPool1d(kernel_size=2, stride=2, padding=2)
        
        self.conv2 = Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.pool2 = MaxPool1d(kernel_size=5, stride=2, padding=2)
        
        self.conv3 = Conv1d(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.pool3 = MaxPool1d(kernel_size=2, stride=2, padding=2)
        self.dropout1 = Dropout(0.2)
        
        self.conv4 = Conv1d(in_channels=256, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool4 = MaxPool1d(kernel_size=2, stride=2, padding=2)
        
        self.flatten = Flatten()
        
        # Calculate the input size for the first fully connected layer
        dummy_input = torch.zeros(1, 1, input_shape)
        dummy_output = self.pool4(self.conv4(self.dropout1(self.pool3(self.conv3(self.pool2(self.conv2(self.pool1(self.conv1(dummy_input)))))))))
        self.fc1_input_size = dummy_output.numel()
        
        self.fc1 = Linear(self.fc1_input_size, 512)
        self.dropout2 = Dropout(0.3)
        
        self.fc2 = Linear(512, 8)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        
        x = F.relu(self.fc2(x), dim=1)
        
        return x
    
def get_model(model_class, dict_path):
    """
    Load a pre-trained model from the specified path.

    Args:
        model_class (class): The class of the model to be loaded.
        dict_path (str): The path to the dictionary containing the model weights.

    Returns:
        model: The loaded model.

    Raises:
        Exception: If the model weights are not found.
    """
    model = model_class()
    try:
        model = torch.load(dict_path)['model']
    except:
        raise Exception("Model weights not found. Please train the model first.")
    return model.to(TrainTesting.device)
    