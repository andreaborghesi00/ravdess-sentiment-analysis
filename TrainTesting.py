import torch
from tqdm import tqdm
import torch.nn as nn
import torchmetrics as tm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emotion_mapping = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}

def validate(model, dl, criterion):
    global device

    model.eval()
    # loss = nn.CrossEntropyLoss()
    acc_metric = tm.Accuracy(task="multiclass", num_classes=8, average='micro').to(device)
    loss_hist = []
    for x, y in dl:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            loss_hist.append(criterion(y_pred, y).item())
            acc_metric.update(torch.argmax(y_pred, dim=1), y)
    model.train()
    return np.mean(loss_hist), acc_metric.compute().cpu().item()

def train(model, train_dl, val_dl, epochs, optimizer, scheduler, criterion):
    global device

    pbar = tqdm(total=epochs*len(train_dl))
    model.train()
    train_acc = tm.Accuracy(task='multiclass', average='micro', num_classes=8).to(device)
    last_val_acc = -1

    val_acc_hist = []
    train_acc_hist = []

    val_loss_hist = []
    train_loss_hist = []
    best_model_state_dict = None

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

            pred = torch.argmax(out, dim=1)
            train_acc.update(pred, y)

            local_train_acc_hist.append(train_acc.compute().cpu().item())
            local_train_loss_hist.append(loss.item())
            train_acc.reset()
            pbar.set_description(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {local_train_acc_hist[-1]:.4f}, val_acc (previous): {last_val_acc:.4f} | best_val_acc: {max(val_acc_hist) if len(val_acc_hist) > 0 else -1:.4f} at epoch {np.argmax(val_acc_hist)+1 if len(val_acc_hist) > 0 else -1}')
            pbar.update(1)

        scheduler.step()
        train_acc_hist.append(np.mean(local_train_acc_hist))
        train_loss_hist.append(np.mean(local_train_loss_hist))

        last_val_loss, last_val_acc = validate(model, val_dl, criterion)
        
        if len(val_acc_hist) == 0 or last_val_acc > max(val_acc_hist):
            best_model_state_dict = model.state_dict()

        val_acc_hist.append(last_val_acc)
        val_loss_hist.append(last_val_loss)

    return train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist, best_model_state_dict

def evaluate_model(net, test_loader):
    global emotion_mapping
    global device
    net.eval()
    gt = []
    pred = []
    
    micro_acc = tm.Accuracy(task='multiclass', average='micro', num_classes=251).to(device)
    macro_acc = tm.Accuracy(task='multiclass', average='macro', num_classes=251).to(device)
    micro_f1_score = tm.F1Score(task='multiclass', average='micro', num_classes=251, multidim_average='global').to(device)
    macro_f1_score = tm.F1Score(task='multiclass', average='macro', num_classes=251, multidim_average='global').to(device)
    micro_precision = tm.Precision(task='multiclass', average='micro', num_classes=251).to(device)
    macro_precision = tm.Precision(task='multiclass', average='macro', num_classes=251).to(device)
    micro_recall = tm.Recall(task='multiclass', average='micro', num_classes=251).to(device)
    macro_recall = tm.Recall(task='multiclass', average='macro', num_classes=251).to(device)

    with torch.no_grad():
        for el, labels in test_loader:
            el = el.to(device)
            labels = labels.to(device)
            predicted = net(el)
            _, predicted = torch.max(predicted, 1)

            micro_acc.update(predicted, labels)
            macro_acc.update(predicted, labels)
            micro_f1_score.update(predicted, labels)
            micro_precision.update(predicted, labels)
            macro_f1_score.update(predicted, labels)
            macro_precision.update(predicted, labels)
            micro_recall.update(predicted, labels)
            macro_recall.update(predicted, labels)
                        
            gt.extend(labels.cpu().numpy())
            pred.extend(predicted.cpu().numpy())
            
    print(f"""
          Micro Accuracy: {micro_acc.compute().item()}\tMacro Accuracy:\t{macro_acc.compute().item()}
          Micro F1 Score: {micro_f1_score.compute().item()}\tMacro F1 Score:\t{macro_f1_score.compute().item()}
          Micro Precision:{micro_precision.compute().item()}\tMacro Precision:{macro_precision.compute().item()}
          Micro Recall:   {micro_recall.compute().item()}\tMacro Recall:\t{macro_recall.compute().item()}
          """)
    
    


