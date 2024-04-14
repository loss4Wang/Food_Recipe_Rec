import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel

import sys
sys.path.append('/local/xiaowang/food_ingredient/')
from utils.utils import get_img_path, set_seeds, EarlyStopping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam

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


# freeze all layers for linear probe
for param in model.parameters():
    param.requires_grad = False

model.eval()

# define labels
label_list = train_explode_df['title'].unique().tolist()
label_list += test_explode_df['title'].unique().tolist()
label_list += val_explode_df['title'].unique().tolist()
num_classes = len(label_list)

tilte2idx = {title: idx for idx, title in enumerate(label_list)}
idx2title = {idx: title for idx, title in enumerate(label_list)}

classifier = torch.nn.Linear(768, num_classes).to(device)


train_pooler_tensor = torch.load('/local/xiaowang/food_ingredient/approach1/train_pooler_output.pt').to(device).squeeze(1)
test_pooler_tensor = torch.load('/local/xiaowang/food_ingredient/approach1/test_pooler_output.pt').to(device).squeeze(1)
val_pooler_tensor = torch.load('/local/xiaowang/food_ingredient/approach1/val_pooler_output.pt').to(device).squeeze(1)



# define dataset and dataloader
class FoodLinearProbeDataset(Dataset):
    def __init__(self, df, img_dir, device):
        self.df = df
        self.img_dir = img_dir
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        title = self.df.iloc[idx]['title']
        class_idx = tilte2idx[title]
        label = F.one_hot(torch.tensor([class_idx]), num_classes=num_classes).float().to(device) # [batch_size, 1, num_classes]

        return idx, label

train_pooler_file = '/local/xiaowang/food_ingredient/approach1/train_pooler_output.pt'
test_pooler_file = '/local/xiaowang/food_ingredient/approach1/test_pooler_output.pt'
val_pooler_file = '/local/xiaowang/food_ingredient/approach1/val_pooler_output.pt'

train_dataset = FoodLinearProbeDataset(train_explode_df, img_dir, device)
test_dataset = FoodLinearProbeDataset(test_explode_df, img_dir,  device)
val_dataset = FoodLinearProbeDataset(val_explode_df, img_dir,  device)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# define optimizer and loss
optimizer = Adam(classifier.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


# Initialize early stopping object
early_stopping = EarlyStopping(patience=3)

# train classifier
for epoch in range(20):
    classifier.train()
    train_loss = 0
    for idx, label in tqdm(train_dataloader):
        vision_output = train_pooler_tensor[idx]
        label = label.squeeze(1)

        optimizer.zero_grad()
        logits = classifier(vision_output)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_dataset)

    # evaluate classifier
    classifier.eval()
    with torch.no_grad():

        val_correct = 0
        val_loss = 0
        for idx, label in tqdm(val_dataloader):
            vision_output = val_pooler_tensor[idx]
            label = label.squeeze(1)
            logits = classifier(vision_output)
            val_correct += (logits.argmax(dim=-1) == label.argmax(dim=-1)).sum().item()
            val_loss += loss_fn(logits, label)
        val_acc = val_correct / len(val_dataset)
        val_loss /= len(val_dataset)

        test_correct = 0
        test_loss = 0
        for idx, label in tqdm(test_dataloader):
            vision_output = test_pooler_tensor[idx]
            label = label.squeeze(1)
            logits = classifier(vision_output)
            test_correct += (logits.argmax(dim=-1) == label.argmax(dim=-1)).sum().item()
            test_loss += loss_fn(logits, label)
        test_acc = test_correct / len(test_dataset)
        test_loss /= len(test_dataset)

    # early stopping
    early_stopping(val_loss=val_loss)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    print(f"Epoch: {epoch+1}, Train Loss:{train_loss}, Val Loss:{val_loss}, Test Acc: {test_acc}, Val Acc: {val_acc}")

# save classifier
save_folder = "/local/xiaowang/food_ingredient/saved_models/approach2/img_classification_classifier"
torch.save(classifier.state_dict(), f"{save_folder}/classifier.pth")