import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel

import sys
sys.path.append('/local/xiaowang/food_ingredient/')
from utils.utils import get_img_path, set_seeds

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam

device = "cuda:6"
batch_size = 128
set_seeds(42)
img_dir = '/local/xiaowang/food_ingredient/1m_test'
train_file = '/local/xiaowang/food_ingredient/Dataset_440_labels/train_set.json'
test_file = '/local/xiaowang/food_ingredient/Dataset_440_labels/test_set.json'
val_file = '/local/xiaowang/food_ingredient/Dataset_440_labels/val_set.json'

train_df = pd.read_json(train_file, orient='records', lines=True)
test_df = pd.read_json(test_file, orient='records', lines=True)
val_df = pd.read_json(val_file, orient='records', lines=True)

train_explode_df = train_df.explode('image_file_name_ls')
test_explode_df = test_df.explode('image_file_name_ls')
val_explode_df = val_df.explode('image_file_name_ls')


model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


# freeze all layers for linear probe
for param in model.parameters():
    param.requires_grad = False

model.eval()

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


# get pooler_output for each image
def get_image_features(img_dir, file_name, model, processor):

    img_path = get_img_path(img_dir, file_name)
    image = Image.open(img_path)

    inputs = processor(images=image, return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model(**inputs)
    vision_output = outputs['pooler_output'] 

    return vision_output.cpu()

train_explode_df['pooler_output'] = train_explode_df.progress_apply(lambda x: get_image_features(img_dir, x['image_file_name_ls'], model, processor), axis=1)
test_explode_df['pooler_output'] = test_explode_df.progress_apply(lambda x: get_image_features(img_dir, x['image_file_name_ls'], model, processor), axis=1)
# val_explode_df['pooler_output'] = val_explode_df.progress_apply(lambda x: get_image_features(img_dir, x['image_file_name_ls'], model, processor), axis=1)

train_pooler_output = torch.stack(train_explode_df['pooler_output'].tolist())
torch.save(train_pooler_output, 'train_pooler_output.pt')

test_pooler_output = torch.stack(test_explode_df['pooler_output'].tolist())
torch.save(test_pooler_output, 'test_pooler_output.pt')

# val_pooler_output = torch.stack(val_explode_df['pooler_output'].tolist())
# torch.save(val_pooler_output, 'val_pooler_output.pt')

print('finish')