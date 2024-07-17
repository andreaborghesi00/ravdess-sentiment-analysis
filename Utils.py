from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import torch
import TrainTesting
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import Models
import TrainTesting
import AudioDatasets

RES_DIR = "results"
PREFIX_CM = "cm"
PREFIX_PLOTS = "loss_acc"
RES_DIR = "results"
PREFIX_MODELS = "model"
PREFIX_ROC = "roc"
PREFIX_PR = "pr"

def _test_probas(model, test_dl):
    """
    Compute the true and predicted values for a given model and test dataloader.

    Args:
        model (torch.nn.Module): The trained model.
        test_dl (torch.utils.data.DataLoader): The test dataloader.

    Returns:
        tuple: A tuple containing the true values and predicted values.
    """
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for el, labels in test_dl:
            el = el.to(TrainTesting.device)
            labels = labels.to(TrainTesting.device)
            predicted = model(el)
            _, predicted = torch.max(predicted, 1)
                        
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return y_true,y_pred

def compute_confusion_matrix(model, test_dl, experiment_name, normalized=True):
    """
    Compute the confusion matrix for a given model and test data.

    Args:
        model (object): The trained model.
        test_dl (DataLoader): The test data loader.
        experiment_name (str): The name of the experiment.
        normalized (bool, optional): Whether to normalize the confusion matrix. Defaults to True.

    Returns:
        None
    """
    global RES_DIR, PREFIX_CM
    
    conf_dir = os.path.join(RES_DIR, "confusion_matrices", model.__class__.__name__)
    os.makedirs(conf_dir, exist_ok=True)

    i = 0 # i know it's ugly, leave me alone
    while os.path.exists(os.path.join(conf_dir, f'{PREFIX_CM}_{experiment_name}_{i}.png')): i += 1

    y_true, y_pred = _test_probas(model, test_dl)
    # y_pred = np.max(y_pred, 1) # thresholding, since the model returns probabilities
    cm = confusion_matrix(y_true, y_pred, normalize='true') if normalized else confusion_matrix(y_true, y_pred)
    # if normalized:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, 2)

    # sns heatmap
    plt.figure(figsize=(15, 20))
    sns.set_theme(font_scale=1.4)
    sns.heatmap(cm, annot=True, fmt='g', cmap='viridis', xticklabels=list(TrainTesting.emotion_mapping.values()), yticklabels=list(TrainTesting.emotion_mapping.values()))
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.savefig(os.path.join(conf_dir, f'{PREFIX_CM}_{experiment_name}_{i}.png'))
    plt.close('all')
    
def plots(model, experiment_name, train_loss, val_loss, train_acc, val_acc):
    """
    Generate and save plots for loss, accuracy, and confusion matrix.

    Args:
        model (object): The trained model object.
        experiment_name (str): The name of the experiment.
        train_loss (list): List of training loss values.
        val_loss (list): List of validation loss values.
        train_acc (list): List of training accuracy values.
        val_acc (list): List of validation accuracy values.
    """
    global RES_DIR, PREFIX_PLOTS
    plot_dir = os.path.join(RES_DIR, "plots", model.__class__.__name__)
    os.makedirs(plot_dir, exist_ok=True)

    i = 0 # i know it's ugly, leave me alone
    while os.path.exists(os.path.join(plot_dir, f'{PREFIX_PLOTS}_{experiment_name}_{i}.png')): i += 1

    # loss and accuracy plots
    plt.figure(figsize=(12, 6))
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
    plt.savefig(os.path.join(plot_dir, f'{PREFIX_PLOTS}_{experiment_name}_{i}.png'))
    plt.close('all')
    
def save_model(model, experiment_name):
    """
    Save the model to disk.

    Args:
        model (object): The trained model object.
        experiment_name (str): The name of the experiment.
    """
    global RES_DIR, PREFIX_MODELS
    model_dir = os.path.join(RES_DIR, "models", model.__class__.__name__)
    os.makedirs(model_dir, exist_ok=True)

    i = 0 # i know it's ugly, leave me alone
    while os.path.exists(os.path.join(model_dir, f'{PREFIX_MODELS}_{experiment_name}_{i}.pth')): i += 1

    torch.save(model.state_dict(), os.path.join(model_dir, f'{PREFIX_MODELS}_{experiment_name}_{i}.pth'))