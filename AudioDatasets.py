import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa
from tqdm import tqdm
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
precomputed_dir = 'numpis'
dataset_dir = 'datasets'
RAVDESS_SUBDIR = 'ravdess'
CREMA_SUBDIR = 'crema/AudioWAV'
SAVEE_SUBDIR = 'savee/ALL'
TESS_SUBDIR = 'tess/TESS Toronto emotional speech set data'
datasets_names = ['ravdess', 'crema', 'savee', 'tess', 'combined']

class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloader(X, y, *args, **kwargs):
    """
    Create and return a DataLoader object for the given dataset.

    Args:
        X (list): The input data.
        y (list): The target labels.
        *args: Additional positional arguments to be passed to the DataLoader constructor.
        **kwargs: Additional keyword arguments to be passed to the DataLoader constructor.

    Returns:
        DataLoader: A DataLoader object for the given dataset.

    """
    dataset = AudioDataset(X, y)
    return DataLoader(dataset, *args, **kwargs)


## Extract path and emotion from the dataset directories

def _get_ravdess():
    """
    Retrieves the RAVDESS dataset.

    Returns:
        audio_emotion (list): List of integers representing the emotions of the audio files.
        audio_path (list): List of strings representing the file paths of the audio files.
    """
    global dataset_dir, RAVDESS_SUBDIR
    ravdess_dir = os.path.join(dataset_dir, RAVDESS_SUBDIR)
    
    audio_emotion = []
    audio_path = []
    for subdir in os.listdir(ravdess_dir):
        actor = os.listdir(os.path.join(ravdess_dir, subdir))
            
        if not str.startswith(subdir, 'Actor_'):
            print('Skipping non-actor directory')
            continue

        for f in actor:
            part = f.split('.')[0].split('-')
            audio_emotion.append(int(part[2])) # the emotion is at the third position of the filename
            audio_path.append(os.path.join(ravdess_dir, subdir, f))
    return audio_emotion, audio_path

def _get_crema():
    """
    Retrieves the audio emotion labels and paths for the CREMA-D dataset.

    Returns:
        audio_emotion (list): A list of strings representing the emotion labels for each audio file.
        audio_path (list): A list of strings representing the file paths for each audio file.
    """
    global dataset_dir, CREMA_SUBDIR
    crema_dir = os.path.join(dataset_dir, CREMA_SUBDIR)
    
    audio_emotion = []
    audio_path = []
    for file in os.listdir(crema_dir):
        audio_path.append(os.path.join(crema_dir, file))

        part=file.split('_')
        if part[2] == 'SAD':
            audio_emotion.append('sad')
        elif part[2] == 'ANG':
            audio_emotion.append('angry')
        elif part[2] == 'DIS':
            audio_emotion.append('disgust')
        elif part[2] == 'FEA':
            audio_emotion.append('fear')
        elif part[2] == 'HAP':
            audio_emotion.append('happy')
        elif part[2] == 'NEU':
            audio_emotion.append('neutral')
        else:
            print('whats this: ', audio_path[-1])
            audio_path.pop()
    return audio_emotion, audio_path

def _get_savee():
    """
    Retrieves the audio emotions and paths for the SAVEE dataset.

    Returns:
        audio_emotion (list): A list of strings representing the emotions of the audio files.
        audio_path (list): A list of strings representing the file paths of the audio files.
    """
    global dataset_dir, SAVEE_SUBDIR
    savee_dir = os.path.join(dataset_dir, SAVEE_SUBDIR)
    
    audio_emotion = []
    audio_path = []
    for file in os.listdir(savee_dir):
        audio_path.append(os.path.join(savee_dir, file))
        part = file.split('_')[1]
        ele = part[:-6]
        if ele=='a':
            audio_emotion.append('angry')
        elif ele=='d':
            audio_emotion.append('disgust')
        elif ele=='f':
            audio_emotion.append('fear')
        elif ele=='h':
            audio_emotion.append('happy')
        elif ele=='n':
            audio_emotion.append('neutral')
        elif ele=='sa':
            audio_emotion.append('sad')
        else:
            audio_emotion.append('surprise')
            
    return audio_emotion, audio_path

def _get_tess():
    """
    Retrieves the audio emotions and paths for the TESS dataset.

    Returns:
        audio_emotion (list): A list of audio emotions.
        audio_path (list): A list of audio paths.
    """
    global dataset_dir, TESS_SUBDIR
    tess_dir = os.path.join(dataset_dir, TESS_SUBDIR)
    
    audio_emotion = []
    audio_path = []
    for dir in os.listdir(tess_dir):
        directories = os.listdir(os.path.join(tess_dir, dir))
        for file in directories:
            audio_path.append(os.path.join(tess_dir, dir, file))
            
            part = file.split('.')[0]
            part = part.split('_')[2]
            if part=='ps':
                audio_emotion.append('surprise')
            else:
                audio_emotion.append(part) # the rest have a proper name
    return audio_emotion, audio_path

def _get_joined_datasets(dataset_names=['crema', 'savee', 'tess']):
    """
    Get the joined datasets for the given dataset names.

    Args:
        dataset_names (list): List of dataset names to be joined. Default is ['crema', 'savee', 'tess'].

    Returns:
        tuple: A tuple containing two lists - audio_emotion and audio_path.
            - audio_emotion (list): List of emotion labels for the audio samples.
            - audio_path (list): List of file paths for the audio samples.

    Raises:
        ValueError: If the dataset name is not one of ['crema', 'savee', 'tess'].

    """
    audio_emotion = []
    audio_path = []
    for dataset_name in dataset_names:
        if dataset_name not in datasets_names:
            raise ValueError(f'Dataset name must be one of {datasets_names}')
        
        datasets_extractors = {
            'crema': _get_crema,
            'savee': _get_savee,
            'tess': _get_tess
        }
        
        audio_emotion_temp, audio_path_temp = datasets_extractors[dataset_name]()
        audio_emotion.extend(audio_emotion_temp)
        audio_path.extend(audio_path_temp)
    return audio_emotion, audio_path

## Data Augmentation

def noise(data, sr):
    """
    Add random noise to the audio data.

    Parameters:
    - data (ndarray): The audio data.
    - sr (int): The sample rate of the audio data.

    Returns:
    - ndarray: The audio data with added noise.
    """
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, sr, rate=0.8):
    """
    Stretch the given audio data by a specified rate.

    Parameters:
    - data (ndarray): The audio data to be stretched.
    - sr (int): The sample rate of the audio data.
    - rate (float, optional): The stretching rate. Default is 0.8.

    Returns:
    - ndarray: The stretched audio data.

    """
    return librosa.effects.time_stretch(y=data, rate=rate)

def shift(data, sr):
    """
    Shifts the given audio data by a random amount within the range of -5 to 5 milliseconds.

    Args:
        data (ndarray): The audio data to be shifted.
        sr (int): The sample rate of the audio data.

    Returns:
        ndarray: The shifted audio data.
    """
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    """
    Applies pitch shifting to the given audio data.

    Parameters:
    - data (ndarray): The audio data to be pitch shifted.
    - sampling_rate (int): The sampling rate of the audio data.
    - pitch_factor (float): The amount of pitch shift to apply. Default is 0.7.

    Returns:
    - ndarray: The pitch-shifted audio data.

    """
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)


## Feature extraction

def extract_features(data, sample_rate):
    """
    Extracts audio features from the given data: zero crossing rate, chroma, mfcc, rms, and mel spectrogram.

    Parameters:
    - data (ndarray): Audio data.
    - sample_rate (int): Sample rate of the audio data.

    Returns:
    - result (ndarray): Extracted audio features.
    """
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result

def get_features(path, augment=False):
    """
    Extracts audio features from the given audio file and eventually augment it.
    The augmentation is done by adding noise, stretching, and pitching the audio data.
    The augmentation produces 4 additional data points for each audio file: shift+noise, stretch+pitch, shift+pitch, stretch+noise.

    Args:
        path (str): The path to the audio file.
        augment (bool, optional): Whether to augment the data with noise, stretching, and pitching. 
                                  Defaults to False.

    Returns:
        numpy.ndarray: An array of extracted audio features.

    """
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)
    
    if augment:
        # data with noise
        temp_data = shift(data, sample_rate)
        noise_data = noise(temp_data, sample_rate)
        res2 = extract_features(noise_data, sample_rate)
        result = np.vstack((result, res2)) # stacking vertically
        
        # data with stretching and pitching
        temp_data = stretch(data, sample_rate)
        data_stretch_pitch = pitch(temp_data, sample_rate)
        res3 = extract_features(data_stretch_pitch, sample_rate)
        result = np.vstack((result, res3)) 

        temp_data = shift(data, sample_rate)
        data_shift_pitch = pitch(temp_data, sample_rate)
        res4 = extract_features(data_shift_pitch, sample_rate)
        result = np.vstack((result, res4))

        temp_data = stretch(data, sample_rate)
        data_stretch_noise = noise(temp_data, sample_rate)
        res5 = extract_features(data_stretch_noise, sample_rate)
        result = np.vstack((result, res5))
    
    return result

def _extract(path, labels, augment=False):
    X_features = []
    Y = []
    pbar = tqdm(total=len(path))
    for idx in range(len(path)):
        feats = get_features(path[idx], augment=augment)
        if augment:
            for f in feats:
                X_features.append(f)
                Y.append(labels[idx])
        else:
            X_features.append(feats)
            Y.append(labels[idx])
        pbar.update()

    X_features = np.array(X_features)
    Y = np.array(Y)
    pbar.close()
    return X_features, Y


def get_data_splits(dataset_name='ravdess', force_recompute=False, random_state=0, train_split=0.7, augment_train=True):
    global precomputed_dir, dataset_dir

    if dataset_name not in datasets_names: raise ValueError(f'Dataset name must be one of {datasets_names}')

    precomputed_req_dir = os.path.join(precomputed_dir, dataset_name)
    os.makedirs(precomputed_req_dir, exist_ok=True)
    
    X_train_path = os.path.join(precomputed_req_dir, 'X_train.npy')
    X_val_path = os.path.join(precomputed_req_dir, 'X_val.npy')
    X_test_path = os.path.join(precomputed_req_dir, 'X_test.npy')
    Y_train_path = os.path.join(precomputed_req_dir, 'Y_train.npy')
    Y_val_path = os.path.join(precomputed_req_dir, 'Y_val.npy')
    Y_test_path = os.path.join(precomputed_req_dir, 'Y_test.npy')
    paths_exist = np.any([os.path.exists(X_train_path), os.path.exists(X_val_path), os.path.exists(X_test_path), os.path.exists(Y_train_path), os.path.exists(Y_val_path), os.path.exists(Y_test_path)])
    
    if force_recompute or paths_exist == False:
        datasets_extractors = {
            'ravdess': _get_ravdess,
            'crema': _get_crema,
            'savee': _get_savee,
            'tess': _get_tess, 
            'combined': _get_joined_datasets
        }
        
        audio_emotion, audio_path = datasets_extractors[dataset_name]()

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(audio_emotion)

        X_train_paths, X_valtest_paths, y_train, y_valtest = train_test_split(audio_path, y, test_size=1-train_split, random_state=random_state, stratify=y)
        X_val_paths, X_test_paths, y_val, y_test = train_test_split(X_valtest_paths, y_valtest, test_size=0.6777, random_state=random_state, stratify=y_valtest)

        X_train, y_train = _extract(X_train_paths, y_train, augment=augment_train)
        X_val, y_val = _extract(X_val_paths, y_val, augment=False)
        X_test, y_test = _extract(X_test_paths, y_test, augment=False)

        np.save(X_train_path, X_train)
        np.save(X_val_path, X_val)
        np.save(X_test_path, X_test)
        np.save(Y_train_path, y_train)
        np.save(Y_val_path, y_val)
        np.save(Y_test_path, y_test)
        
    else:
        print('Loading precomputed data splits')
        X_train = np.load(X_train_path)
        X_val = np.load(X_val_path)
        X_test = np.load(X_test_path)
        y_train = np.load(Y_train_path)
        y_val = np.load(Y_val_path)
        y_test = np.load(Y_test_path)
    return X_train, X_val, X_test, y_train, y_val, y_test




        
