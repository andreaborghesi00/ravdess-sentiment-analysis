import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa
from tqdm import tqdm
import os

precomputed_dir = 'precomputed_datasets'

class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class AugmentedAudioDataset(Dataset):
    def __init__(self, X, y, augmentation_prob=0.5, transforms=None):
        self.X = X
        self.y = y
        self.y = torch.tensor(self.y, dtype=torch.long)
        self.augmentation_prob = augmentation_prob
        self.transforms = transforms
    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        feats = get_features(self.X[idx], augmentation_prob=self.augmentation_prob, transform=self.transforms)
        # check shape, if not correct, add padding
        # if feats.shape[0] != 108:
        #     feats = np.hstack((feats, np.zeros((feats.shape[0], 108-feats.shape[1]))))

        feats = torch.tensor(feats, dtype=torch.float32)
        feats = feats.unsqueeze(0)
        return feats, self.y[idx]


class PreAugmentedDataset(Dataset):
    def __init__(self, X, y, augmentation_prob=0.8, dataset_multiplier=4, transforms=None):
        self.X = X
        self.y = y
        self.y = torch.tensor(self.y, dtype=torch.long)
        self.augmentation_prob = augmentation_prob
        self.transforms = transforms

        self.X_features = []

        for _ in range(dataset_multiplier):
            for idx in tqdm(range(len(self.X))):
                feats = get_features(self.X[idx], augmentation_prob=self.augmentation_prob, transform=self.transforms)
                self.X_features.append(feats)
        self.X_features = torch.tensor(self.X_features, dtype=torch.float32)
        self.X_features = self.X_features.unsqueeze(1)
        self.y = self.y.repeat(dataset_multiplier, 1)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X_features[idx], self.y[idx]
    

def noise(data, sr):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data


def stretch(data, sr, rate=0.8):
    return librosa.effects.time_stretch(y=data, rate=rate)


def shift(data, sr):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)


def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def get_features(path, augment=False):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)
    
    if augment:
        # data with noise
        noise_data = noise(data, sample_rate)
        res2 = extract_features(noise_data, sample_rate)
        result = np.vstack((result, res2)) # stacking vertically
        
        # data with stretching and pitching
        new_data = stretch(data, sample_rate)
        data_stretch_pitch = pitch(new_data, sample_rate)
        res3 = extract_features(data_stretch_pitch, sample_rate)
        result = np.vstack((result, res3)) # stacking vertically
    
    return result


def augment_data(X, y, augmentation_prob=0.5, transform=None, dataset_multiplier=2, save_path=None, force_recompute=False, normalizer=None, train=False):
    global precomputed_dir
    X_features = []

    if not force_recompute and os.path.exists(os.path.join(precomputed_dir, save_path)):
        print(f"Loading precomputed data from {save_path}")
        X_features = np.load(os.path.join(precomputed_dir, save_path))
        y = np.array(y)
        y = np.repeat(y, dataset_multiplier, axis=0)
        return X_features, y, normalizer
    
    print("Computing features...")
    for m in range(dataset_multiplier):
        print(f"Augmentation {m+1}/{dataset_multiplier}")
        for idx in tqdm(range(len(X))):
            feats = get_features(X[idx], augmentation_prob=augmentation_prob, transform=transform)
            X_features.append(feats)
    X_features = np.array(X_features)
    if train and normalizer is not None:
        X_features = normalizer.fit_transform(X_features)
    elif normalizer is not None:
        X_features = normalizer.transform(X_features)

    y = np.array(y)
    print(f"Saving precomputed data at {os.path.join(precomputed_dir, save_path)}")
    if save_path != None:
        np.save(os.path.join(precomputed_dir, save_path), X_features)
    return X_features, y, normalizer

        
