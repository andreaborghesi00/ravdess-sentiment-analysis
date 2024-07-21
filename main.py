import AudioDatasets
import Models
import torch
import TrainTesting
import Utils

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import numpy as np
import os
import gc
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import logging

emotion_mapping = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}
pretrain_datasets = [None, 'crema', 'savee', 'combined', 'tess']
pretrain_models_root = 'pretrained_models'
scalers_root = 'scalers'
pretraining = False
force_recompute = False
random_state = 0
train_split = 0.7


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.info(f"Starting... | pretraining: {pretraining} | pretrain_datasets: {pretrain_datasets} | force_recompute: {force_recompute}")

if __name__ == '__main__':
    for dataset_name in pretrain_datasets:
        if pretraining and dataset_name is None: continue
        
        print(f'### {str.upper(dataset_name if dataset_name is not None else "RAW")} ###')
        experiment_name = f'{dataset_name}_pretrained' if dataset_name is not None else 'raw'
        
        training_dataset = dataset_name if pretraining else "ravdess"
        logger.debug(f"training_dataset: {str.upper(training_dataset)}")
        logger.info(f"Loading {training_dataset} dataset...")
        X_train, X_val, X_test, y_train, y_val, y_test = AudioDatasets.get_data_splits(dataset_name=training_dataset, force_recompute=force_recompute, augment_train=True, random_state=random_state, train_split=train_split)
        
        # beware, if you change the split, the scaler has to be recomputed, otherwise data will be leaked, i'll commit to a 70:10:20 seeded split everywhere for now
        scaler_path = os.path.join(scalers_root, f'{training_dataset}_scaler_seed{random_state}_tsplit{int(train_split*100)}.pkl')
        if not os.path.exists(scaler_path):
            logger.info(f"Scaler not found, creating new scaler...")
            scaler = StandardScaler() # Standardize features
        else:
            scaler = pickle.load(open(scaler_path, 'rb'))
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # save scaler
        if not os.path.exists(scaler_path):
            pickle.dump(scaler, open(scaler_path, 'wb'))

        # Add channel dimension
        X_train = np.expand_dims(X_train, axis=1)
        X_val = np.expand_dims(X_val, axis=1)
        X_test = np.expand_dims(X_test, axis=1)

        # Create dataloaders
        logger.info(f"Creating dataloaders...")
        batch_size = 64

        train_dl = AudioDatasets.get_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False, prefetch_factor=2, persistent_workers=True)
        val_dl = AudioDatasets.get_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        test_dl = AudioDatasets.get_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        
        # Create model
        logger.info(f"Creating model...")
        # model = Models.AudioCNN()
        logger.debug(f"X_train.shape: {X_train.shape}")
        model = Models.AudioLSTM(input_size=X_train.shape[2], hidden_size=256, num_layers=5, log_level=logging.WARNING)
        model = model.to(TrainTesting.device)
        if not pretraining and dataset_name is not None:
            logger.info(f"Loading weights...")
            model.load_state_dict(torch.load(os.path.join(pretrain_models_root, f'{model.__class__.__name__}_{dataset_name}.pth')))
        
        epochs = 80
        optimizer = AdamW(model.parameters(), lr=5e-5, fused=True, weight_decay=1e-6)
        # scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=1e-6)
        criterion = torch.nn.CrossEntropyLoss()

        # Train model
        logger.info(f"Training model...")
        train_loss, val_loss, train_acc, val_acc, best_dict = TrainTesting.train(model, train_dl, val_dl, epochs, optimizer, scheduler, criterion)
        TrainTesting.evaluate_model(model, test_dl)
        
        ### PRETRAINING ONLY ###
        if pretraining:
            torch.save(model.state_dict(), os.path.join(pretrain_models_root, f'{model.__class__.__name__}_{dataset_name}.pth'))      
        else:
        ### POST-PRETRAINING ONLY ###
            Utils.save_model(model, experiment_name)
            Utils.plots(model, experiment_name, train_loss, val_loss, train_acc, val_acc)
            Utils.compute_confusion_matrix(model, test_dl, experiment_name, normalized=True)
            Utils.save_numpis(model, experiment_name, train_loss, val_loss, train_acc, val_acc)
        
        # clear memory
        del X_train, X_val, X_test, y_train, y_val, y_test, train_dl, val_dl, test_dl, model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()
        gc.collect()