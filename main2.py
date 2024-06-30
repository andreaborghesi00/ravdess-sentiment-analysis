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


root_datasets = 'precomputed_datasets'
emotion_mapping = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}

if __name__ == '__main__':
    # Load data
    print('Loading data...')

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

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(audio_emotion)

    X_train_paths, X_valtest_paths, y_train, y_valtest = train_test_split(audio_path, y, test_size=0.2, random_state=42, stratify=y)
    X_val_paths, X_test_paths, y_val, y_test = train_test_split(X_valtest_paths, y_valtest, test_size=0.5, random_state=42, stratify=y_valtest)
    
    X_train, y_train = AudioDatasets.augment_data(X_train_paths, y_train, augmentation_prob=0.8, transform=[stretch, shift, pitch, noise], dataset_multiplier=4, save_path='train_augmented_70x4.npy', force_recompute=False)
    X_val, y_val = AudioDatasets.augment_data(X_val_paths, y_val, augmentation_prob=0, transform=None, dataset_multiplier=1, save_path='val_10.npy', force_recompute=False)
    X_test, y_test = AudioDatasets.augment_data(X_test_paths, y_test, augmentation_prob=0, transform=None, dataset_multiplier=1, save_path='test_20.npy', force_recompute=False)

    # Create datasets
    print('Creating datasets...')
    train_ds = AudioDatasets.AudioDataset(X_train, y_train)
    val_ds = AudioDatasets.AudioDataset(X_val, y_val)
    test_ds = AudioDatasets.AudioDataset(X_test, y_test)
    print(f"Lengths: Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

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
    model = Models.SimpleAudioCNN()
    model = model.to(TrainTesting.device)
    print(f"N. of parameters: {sum(p.numel() for p in model.parameters())}")

    # Create optimizer
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Create scheduler
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    # scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-8)
    scheduler = None
    # Create criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Train model
    epochs = 10
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
    test_loss, test_acc = TrainTesting.validate(model, test_dl)
    print(f'Test loss: {test_loss}, Test acc: {test_acc}')

    TrainTesting.evaluate_model(model, test_dl)