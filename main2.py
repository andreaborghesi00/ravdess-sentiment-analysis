import AudioDatasets
import Models
import torch
import TrainTesting
from AudioDatasets import noise, stretch, shift, pitch

from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import numpy as np
import os

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torchmetrics as tm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

root_datasets = 'precomputed_datasets'
emotion_mapping = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}

if __name__ == '__main__':
    # Load data
    print('Loading data...')
    force_recompute = False
    if force_recompute or not (os.path.exists('X_features.npy') and os.path.exists('Y.npy')):
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
                audio_emotion.append(int(part[2])) # the emotion is at the third position of the filename
                audio_path.append(os.path.join(dataset_path, subdir, f))

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(audio_emotion)

        # ohe = OneHotEncoder()
        # y = ohe.fit_transform(np.array(audio_emotion).reshape(-1, 1)).toarray()

        X_train_paths, X_valtest_paths, y_train, y_valtest = train_test_split(audio_path, y, test_size=0.4, random_state=42, stratify=y)
        X_val_paths, X_test_paths, y_val, y_test = train_test_split(X_valtest_paths, y_valtest, test_size=0.5, random_state=42, stratify=y_valtest)

        def extract(path, labels, augment=False):
            X_features = []
            Y = []
            pbar = tqdm(total=len(path))
            for idx in range(len(path)):
                feats = AudioDatasets.get_features(path[idx], augment=augment)
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

        # X, Y = extract(audio_path, y, augment=False)
        # np.save('X_non_augmented.npy', X)
        # np.save('Y_non_augmented.npy', Y)

        X_val, y_val = extract(X_val_paths, y_val, augment=False)
        X_train, y_train = extract(X_train_paths, y_train, augment=True)
        X_test, y_test = extract(X_test_paths, y_test, augment=False)

        np.save('numpis/X_train.npy', X_train)
        np.save('numpis/X_val.npy', X_val)
        np.save('numpis/X_test.npy', X_test)
        np.save('numpis/Y_train.npy', y_train)
        np.save('numpis/Y_val.npy', y_val)
        np.save('numpis/Y_test.npy', y_test)
    else:
        X_train = np.load('numpis/X_train.npy')
        X_val = np.load('numpis/X_val.npy')
        X_test = np.load('numpis/X_test.npy')
        y_train = np.load('numpis/Y_train.npy')
        y_val = np.load('numpis/Y_val.npy')
        y_test = np.load('numpis/Y_test.npy')

    #     X = np.load('X_non_augmented.npy')
    #     Y = np.load('Y_non_augmented.npy')

    # X_train, X_valtest, y_train, y_valtest = train_test_split(X, Y, test_size=0.4, random_state=42, stratify=Y)
    # X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42, stratify=y_valtest)

    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    
    # X_train, y_train, normalizer = AudioDatasets.augment_data(X_train_paths, y_train, augmentation_prob=0.8, transform=[stretch, shift, pitch, noise], dataset_multiplier=4, save_path='train_augmented_70x4.npy', force_recompute=False, normalizer=normalizer, train=True)
    # X_val, y_val, normalizer = AudioDatasets.augment_data(X_val_paths, y_val, augmentation_prob=0, transform=None, dataset_multiplier=1, save_path='val_10.npy', force_recompute=False, normalizer=normalizer, train=False)
    # X_test, y_test, normalizer = AudioDatasets.augment_data(X_test_paths, y_test, augmentation_prob=0, transform=None, dataset_multiplier=1, save_path='test_20.npy', force_recompute=False, normalizer=normalizer, train=False)

    # Create datasets
    print('Creating datasets...')
    train_ds = AudioDatasets.AudioDataset(X_train, y_train)
    val_ds = AudioDatasets.AudioDataset(X_val, y_val)
    test_ds = AudioDatasets.AudioDataset(X_test, y_test)
    print(f"Lengths: Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    sample_x, sample_y = train_ds.__getitem__(0)
    print(f"Sample x shape: {sample_x.shape}, Sample y shape: {sample_y.shape}")

    # Create dataloaders
    print('Creating dataloaders...')
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Create model
    input_size = X_train.shape[1]
    hidden_size = 512
    num_layers = 6

    print('Creating model...')
    # model = Models.AudioLSTM(input_size, hidden_size, num_layers)
    model = Models.AudioCNN()
    model = model.to(TrainTesting.device)
    print(f"N. of parameters: {sum(p.numel() for p in model.parameters())}")

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5, fused=True, weight_decay=1e-6)

    # Create scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    # scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-8)
    # scheduler = None
    # Create criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Train model
    epochs = 150
    print('Training model...')
    train_loss, val_loss, train_acc, val_acc = TrainTesting.train(model, train_dl, val_dl, epochs, optimizer, scheduler, criterion)
    
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
    plt.savefig('last.png')


    # Test model
    print('Testing model...')
    test_loss, test_acc = TrainTesting.validate(model, test_dl, criterion)
    print(f'Test loss: {test_loss}, Test acc: {test_acc}')

    TrainTesting.evaluate_model(model, test_dl)