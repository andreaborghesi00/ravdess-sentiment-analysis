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
    force_recompute = True
    
    X_train, X_val, X_test, y_train, y_val, y_test = AudioDatasets.get_data_splits(force_recompute=force_recompute)
    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    print(f'X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')
    print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
    scaler = StandardScaler() # Standardize features
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Add channel dimension, required for CNN
    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    # Create dataloaders
    print('Creating dataloaders...')
    batch_size = 64

    train_dl = AudioDatasets.get_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False, prefetch_factor=2, persistent_workers=True)
    val_dl = AudioDatasets.get_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    test_dl = AudioDatasets.get_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    
    # Create model
    input_size = X_train.shape[1]
    hidden_size = 512
    num_layers = 6

    print('Creating model...')
    # model = Models.AudioLSTM(input_size, hidden_size, num_layers)
    model = Models.AudioCNN()
    model = model.to(TrainTesting.device)
    print(f"N. of parameters: {sum(p.numel() for p in model.parameters())}")

    epochs = 60
    optimizer = AdamW(model.parameters(), lr=5e-5, fused=True, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = torch.nn.CrossEntropyLoss()

    # Train model
    print('Training model...')
    train_loss, val_loss, train_acc, val_acc, best_dict = TrainTesting.train(model, train_dl, val_dl, epochs, optimizer, scheduler, criterion)
    
    print('Testing last model...')
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

    print('Testing best model...')
    model.load_state_dict(best_dict)
    test_loss, test_acc = TrainTesting.validate(model, test_dl, criterion)
    print(f'Test loss: {test_loss}, Test acc: {test_acc}')

    TrainTesting.evaluate_model(model, test_dl)