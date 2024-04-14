import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import numpy as np

from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel

import os
import sys
sys.path.append('/local/xiaowang/food_ingredient/')
from utils.utils import get_img_path, set_seeds, EarlyStopping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score

device = "cuda:7"
batch_size = 128
set_seeds(42)
img_dir = '/local/xiaowang/food_ingredient/1m_test'
train_file = '/local/xiaowang/food_ingredient/Dataset_440_labels/train_set.json'
test_file = '/local/xiaowang/food_ingredient/Dataset_440_labels/test_set.json'
val_file = '/local/xiaowang/food_ingredient/Dataset_440_labels/val_set.json'

train_df = pd.read_json(train_file, orient='records', lines=True)
test_df = pd.read_json(test_file, orient='records', lines=True)
val_df = pd.read_json(val_file, orient='records', lines=True)

train_explode_df = train_df.explode('image_file_name_ls', ignore_index=True)
test_explode_df = test_df.explode('image_file_name_ls', ignore_index=True)
val_explode_df = val_df.explode('image_file_name_ls', ignore_index=True)

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.eval()

# freeze all layers for linear probe
for param in model.parameters():
    param.requires_grad = False

# define labels
label_list = train_explode_df.explode('cleaned_ingredients').cleaned_ingredients.unique().tolist()
label_list += test_explode_df.explode('cleaned_ingredients').cleaned_ingredients.unique().tolist()
label_list += val_explode_df.explode('cleaned_ingredients').cleaned_ingredients.unique().tolist()
label_list = list(set(label_list))
num_classes = len(label_list)
print(num_classes)

label2idx = {label: idx for idx, label in enumerate(label_list)}
idx2label = {idx: label for idx, label in enumerate(label_list)}

classifier = torch.nn.Linear(768, num_classes).to(device)

train_pooler_tensor = torch.load('train_pooler_output.pt').to(device).squeeze(1)
val_pooler_tensor = torch.load('val_pooler_output.pt').to(device).squeeze(1)
test_pooler_tensor = torch.load('test_pooler_output.pt').to(device).squeeze(1)

# define dataset and dataloader
class FoodLinearProbeDataset(Dataset):
    def __init__(self, df, img_dir, device):
        self.df = df
        self.img_dir = img_dir
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        labels = self.df.iloc[idx]['cleaned_ingredients']
        class_idx = [label2idx[label] for label in labels]
        ohe = F.one_hot(torch.tensor(class_idx), num_classes=num_classes).sum(0).float().to(device)

        return idx,ohe 

loss_fn = nn.BCEWithLogitsLoss()
optimizer = Adam(classifier.parameters(), lr=1e-3)

train_dataset = FoodLinearProbeDataset(train_explode_df, img_dir, device)
val_dataset = FoodLinearProbeDataset(val_explode_df, img_dir, device)
test_dataset = FoodLinearProbeDataset(test_explode_df, img_dir, device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize early stopping object
early_stopping = EarlyStopping(patience=3)

# train classifier
for epoch in range(20):
    classifier.train()
    train_loss = 0
    for idx, label in tqdm(train_loader):
        # forward pass
        vision_output = train_pooler_tensor[idx]

        # backward pass
        optimizer.zero_grad()
        logits = classifier(vision_output)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_dataset)

    print(f"Epoch: {epoch+1}, Train Loss:{train_loss}")

    # evaluate classifier
    classifier.eval()
    all_preds = []  # List to store all predictions
    all_labels = []  # List to store all labels
    with torch.no_grad():
        val_loss = 0
        for idx, label in tqdm(val_loader):
            vision_output = val_pooler_tensor[idx]
            logits = classifier(vision_output)
            val_loss += loss_fn(logits, label).item()

            batch_pred_np = logits.detach().cpu().numpy()
            batch_pred_np = (batch_pred_np > 0.5).astype(int) # threshold 0.5
            batch_ohe_np = label.detach().cpu().numpy().astype(int)

            all_preds.append(batch_pred_np)
            all_labels.append(batch_ohe_np)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)   

    val_loss /= len(val_dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_mi_f1 = f1_score(all_labels, all_preds, average='micro')
    val_miprecision = precision_score(all_labels, all_preds, average='micro')
    val_mi_recall = recall_score(all_labels, all_preds, average='micro')
    val_ma_f1 = f1_score(all_labels, all_preds, average='macro')
    val_ma_precision = precision_score(all_labels, all_preds, average='macro')
    val_ma_recall = recall_score(all_labels, all_preds, average='macro')

    print(f"Epoch: {epoch+1}, Train Loss:{train_loss}, Val Loss:{val_loss}, Val Acc: {val_acc}, Val Micro F1: {val_mi_f1}, Val Micro Precision: {val_miprecision}, Val Micro Recall: {val_mi_recall}, Val Macro F1: {val_ma_f1}, Val Macro Precision: {val_ma_precision}, Val Macro Recall: {val_ma_recall}")

    # evaluate on test set
    all_preds = []  # List to store all predictions
    all_labels = []  # List to store all labels
    with torch.no_grad():
        test_loss = 0
        for idx, label in tqdm(test_loader):
            vision_output = test_pooler_tensor[idx]
            logits = classifier(vision_output)
            test_loss += loss_fn(logits, label).item()

            batch_pred_np = logits.detach().cpu().numpy()
            batch_pred_np = (batch_pred_np > 0.5).astype(int) # threshold 0.5
            batch_ohe_np = label.detach().cpu().numpy().astype(int)

            all_preds.append(batch_pred_np)
            all_labels.append(batch_ohe_np)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    test_loss /= len(test_dataset)
    test_acc = accuracy_score(all_labels, all_preds)
    test_mi_f1 = f1_score(all_labels, all_preds, average='micro')
    test_miprecision = precision_score(all_labels, all_preds, average='micro')
    test_mi_recall = recall_score(all_labels, all_preds, average='micro')
    test_ma_f1 = f1_score(all_labels, all_preds, average='macro')
    test_ma_precision = precision_score(all_labels, all_preds, average='macro')
    test_ma_recall = recall_score(all_labels, all_preds, average='macro')

    print(f"Epoch: {epoch+1}, Train Loss:{train_loss}, Val Loss:{val_loss}, Test Acc: {test_acc}, Test Micro F1: {test_mi_f1}, Test Micro Precision: {test_miprecision}, Test Micro Recall: {test_mi_recall}, Test Macro F1: {test_ma_f1}, Test Macro Precision: {test_ma_precision}, Test Macro Recall: {test_ma_recall}")

    # early stopping
    early_stopping(val_loss=val_loss)
    if early_stopping.early_stop:
        print("Early stopping")
        break


# save classifier
save_folder = "/local/xiaowang/food_ingredient/saved_models/approach1/multi_label_classifier"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

torch.save(classifier.state_dict(), f"{save_folder}/classifier.pth")