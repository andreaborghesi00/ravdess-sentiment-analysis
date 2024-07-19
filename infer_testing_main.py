import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import AudioDatasets
import Models
import TrainTesting
import Utils
import pickle
import numpy as np
# Define the path to the pretrained model
model_path = 'results/models/AudioCNN/model_savee_pretrained_1.pth'

# Create an instance of your model
model = Models.get_model(Models.AudioCNN, model_path)

_, _, X_test, _, _, y_test = AudioDatasets.get_data_splits(dataset_name='ravdess', force_recompute=False, augment_train=True, random_state=0, train_split=0.7)

scaler = pickle.load(open('ravdess_scaler.pkl', 'rb'))

X_test = scaler.transform(X_test)
X_test = np.expand_dims(X_test, axis=1)

test_dl = AudioDatasets.get_dataloader(X_test, y_test, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

TrainTesting.evaluate_model(model, test_dl)
