# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Importing Libraries

# %%
# arrays and math
import numpy as np
from scipy import fftpack

# data handling
import pickle
import os
import albumentations as albus
from albumentations.pytorch import ToTensorV2
import IPython.display as ipd
from IPython.display import Audio

# fits
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.visualization import simple_norm


# plotting
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, BatchNorm1d, Dropout1d, ReLU, Sigmoid, GELU, Module, Sequential, Linear
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import torchmetrics as tm
from torchsummary import summary

# sklearn (for machine learning)
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder 

# feature extraction
import librosa
import librosa.display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



# %% [markdown]
# File naming convention
#
# Each of the 1440 files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 03-01-06-01-02-01-12.wav). These identifiers define the stimulus characteristics:
#
# Filename identifiers
#
# Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
#
# Vocal channel (01 = speech, 02 = song).
#
# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
#
# Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
#
# Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
#
# Repetition (01 = 1st repetition, 02 = 2nd repetition).
#
# Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
#
# Filename example: 03-01-06-01-02-01-12.wav
#
# Audio-only (03)
# Speech (01)
# Fearful (06)
# Normal intensity (01)
# Statement "dogs" (02)
# 1st Repetition (01)
# 12th Actor (12)
# Female, as the actor ID number is even.

# %%
audio_emotion = []
audio_path = []
dataset_path = "dataset"
for subdir in os.listdir(dataset_path):
    actor = os.listdir(os.path.join(dataset_path, subdir))
    
    if not str.startswith(subdir, 'Actor_'):
        print('Skipping non-actor directory')
        continue

    for f in actor:
        part = f.split('.')[0].split('-')
        # print(int(part[2]))
        audio_emotion.append(int(part[2])) # the emotion is at the third position of the filename
        audio_path.append(os.path.join(dataset_path, subdir, f))

# %%
emotion_mapping = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}

# %%
# plot emotion count
plt.figure(figsize=(10, 5))
plt.bar(emotion_mapping.values(), [audio_emotion.count(i) for i in range(1, 9)], color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray'])
plt.title('Emotion count')
plt.show()


# %%
data, sr = librosa.load(audio_path[41])

# %%
# CREATE LOG MEL SPECTROGRAM
plt.figure(figsize=(10, 5))
spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128,fmax=8000) 
log_spectrogram = librosa.power_to_db(spectrogram)
librosa.display.specshow(log_spectrogram, y_axis='mel', sr=sr, x_axis='time');
plt.title('Mel Spectrogram ')
plt.colorbar(format='%+2.0f dB')
plt.show()

# %%
mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=30)


# MFCC
plt.figure(figsize=(10, 10))
plt.subplot(3,1,1)
librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()
plt.show()


# %% [markdown]
# # Feature Extraction

# %%
def get_features(file_path, frame_length=2048, hop_length=512, n_mfcc=30, n_fft=2048, n_mels=128, transform=None, augmentation_prob=0.5):
    data, sr=librosa.load(file_path, duration=2.5, offset=0.6)

    if transform:
        for i in range(len(transform)):
            if np.random.uniform(0, 1) < augmentation_prob:
                data = transform[i](data, sr)

    feats = np.array([[]])
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    zcr = np.squeeze(zcr)[:, np.newaxis]
    rmse=librosa.feature.rms(y=data,frame_length=frame_length, hop_length=hop_length)
    rmse = np.squeeze(rmse)[:, np.newaxis]
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    mfcc = np.squeeze(mfcc.T)

    feats = np.hstack((zcr, rmse, mfcc))
    feats = np.moveaxis(feats, 1, 0)
    return feats


# %%
# %%time 
# check the time, i'm deciding if i should extract features while training to apply data augmentation online and always have fresh data
feats = get_features(audio_path[41])
# print(feats.shape)

# %%
print(feats.shape)


# %%
# X, y = [], []

# for audio, emo, idx in tqdm(zip(audio_path, audio_emotion, range(len(audio_path))), total=len(audio_path)):
#     try:
#         feats = get_features(audio)
#         X.append(feats)
#         y.append(emo)
#     except:
#         print(f'Error in file {audio}')

# %% [markdown]
# # Data augmentation

# %%
def noise(data, sr):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, sr, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data, sr):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


# %% [markdown]
# # Data Preparation

# %%
# one hot encode the labels
ohe = OneHotEncoder()
y = ohe.fit_transform(np.array(audio_emotion).reshape(-1, 1)).toarray()
y.shape, len(audio_path)

# %%
X_train, X_valtest, y_train, y_valtest = train_test_split(audio_path, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5)


# %%
class AudioDataset(Dataset):
    def __init__(self, X, y, augmentation_prob=0.5, transforms=None):
        self.X = X
        self.y = y
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.augmentation_prob = augmentation_prob
        self.transforms = transforms
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feats = get_features(self.X[idx], augmentation_prob=self.augmentation_prob, transform=self.transforms)
        # check shape, if not correct, add padding
        if feats.shape[1] != 108:
            feats = np.hstack((feats, np.zeros((feats.shape[0], 108-feats.shape[1]))))

        feats = torch.tensor(feats, dtype=torch.float32)
        return feats, self.y[idx]


# %%
transforms = [noise, stretch, shift, pitch]

train_ds = AudioDataset(X_train, y_train)
val_ds = AudioDataset(X_val, y_val)
test_ds = AudioDataset(X_test, y_test)
print(train_ds.__getitem__(23)[0].shape)

# %%
batch_size = 64
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


# %%
class AudioCNN(Module):
    def __init__(self): # in size: 108
        super(AudioCNN, self).__init__()
        self.conv1 = Sequential(
            Conv1d(22, 64, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(64),
            GELU(),
            Conv1d(64, 64, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(64),
            GELU(),
            MaxPool1d(kernel_size=2, stride=2)
        ) # out size: 54
        self.conv2 = Sequential(
            Conv1d(64, 128, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(128),
            GELU(),
            Conv1d(128, 128, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(128),
            GELU(),
            MaxPool1d(kernel_size=2, stride=2)
        ) # out size: 27
        self.conv3 = Sequential(
            Conv1d(128, 256, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(256),
            GELU(),
            Conv1d(256, 256, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(256),
            GELU(),
            MaxPool1d(kernel_size=2, stride=2)
        ) # out size: 13
        self.conv4 = Sequential(
            Conv1d(256, 512, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(512),
            GELU(),
            Conv1d(512, 512, kernel_size=3, stride=1, padding='same'),
            BatchNorm1d(512),
            GELU(),
            MaxPool1d(kernel_size=2, stride=2)
        ) # out size: 6

        self.fc1 = Sequential(
            Linear(512*6, 1024),
            GELU(),
            Dropout1d(0.2),
            Linear(1024, 512),
            GELU(),
            Dropout1d(0.2),
            Linear(512, 8),
            Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x



# %%
def validate(model, dl):
    model.eval()
    loss = nn.BCELoss()
    acc_metric = tm.Accuracy(task="multiclass", num_classes=8, average='micro').to(device)
    loss_hist = []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss_hist.append(loss(y_pred, y).item())
        acc_metric.update(y_pred, y)
    
    return np.mean(loss_hist), acc_metric.compute()


# %%
def train(model, train_dl, val_dl, epochs, optimizer, scheduler, criterion):
    pbar = tqdm(total=epochs*len(train_dl))
    model.train()
    train_acc = tm.Accuracy(task='multiclass', average='micro', num_classes=8).to(device)
    last_val_acc = -1

    val_acc_hist = []
    train_acc_hist = []

    val_loss_hist = []
    train_loss_hist = []
    train_acc 
    for epoch in range(epochs):
        local_train_acc_hist = []
        local_train_loss_hist = []
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_acc.update(out, y)
            local_train_acc_hist.append(train_acc.compute().item())
            local_train_loss_hist.append(loss.item())
            pbar.set_description(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {local_train_acc_hist[-1]:.4f}, val_acc (previous): {last_val_acc:.4f} | best_val_acc: {max(val_acc_hist) if len(val_acc_hist) > 0 else -1:.4f} at epoch {np.argmax(val_acc_hist)+1 if len(val_acc_hist) > 0 else -1}')
            pbar.update(1)
        train_acc_hist.append(np.mean(local_train_acc_hist))

        last_val_acc, last_val_loss = validate(model, val_dl)
        val_acc_hist.append(last_val_acc)
        val_loss_hist.append(last_val_loss)
    return train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist


# %%
experiment_name = 'cnn'
model = AudioCNN().to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
criterion = nn.BCELoss()
epochs = 30

# %%
summary(model, (22, 108));

# %%
(y.shape == (1440, 8))

# %%
train_loss, val_loss, train_acc, val_acc = train(model, train_dl, val_dl, epochs, optimizer, scheduler, criterion)

# save model
i = 0
while os.path.exists(f'results/{experiment_name}_{i}'): i += 1

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'train_acc': train_acc,
    'val_acc': val_acc
}, f'results/{experiment_name}_{i}.pth')

# %%
# pretty plots
val_loss = val_loss.c

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='train')
plt.plot(val_loss, label='val')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='train')
plt.plot(val_acc, label='val')
plt.title('Accuracy')
plt.legend()
plt.show()

# %%
validate(model, test_dl)
