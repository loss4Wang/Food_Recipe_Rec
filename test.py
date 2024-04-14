#!/usr/bin/env python
# coding: utf-8

import io
import pickle
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

model_path = './model_train_epoch_10.pth'
MLB_PKL = './mlb.pkl'
BATCH_SIZE = 16
THRESHOLD = 0.35
num_unique_ingredients = 440
model = models.resnet50(pretrained=True)
num_in_features = model.fc.in_features
model.fc = nn.Linear(num_in_features, num_unique_ingredients)
model.load_state_dict(torch.load(model_path))

criterion = nn.BCEWithLogitsLoss()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


with open(MLB_PKL, 'rb') as file:
    mlb = pickle.load(file)

def test(img_path, threshold =THRESHOLD):
    model.eval()
    image = Image.open(img_path).convert('RGB')
    image = preprocess(image)
    img_tensor = image.unsqueeze(0)
    outputs = model(img_tensor)
    outputs = torch.sigmoid(outputs)
    targets_idx = [[idx for idx, col in enumerate(row) if col >= threshold] for row in outputs.cpu()]
    return [mlb.classes_[idx] for idx in targets_idx[0]]

def st_show():
    st.title("Ingredients Recognition with Fine-tuned Resnet50")
    uploaded_file = st.file_uploader("Choose an food image:", type={"jpg", "jpeg", "png"})
    if uploaded_file is not None:
        print(uploaded_file, type(uploaded_file))
        #image = Image.open(io.BytesIO(uploaded_file.read()))
        image = Image.open(uploaded_file)
        st.image(image, caption='Your food image', use_column_width=True)
        threshold = st.slider('Choose your threshold:', min_value=0.1, max_value=0.6, value=0.35, step=0.05)
        if st.button('Predict'):
            ingredients = test(uploaded_file, threshold=threshold)
            if st.get_option("theme.primaryColor") == "#000000":
            # light
                color_style = "color: green;"
            else:
            # dark
                color_style = "color: yellow;"
            ingredient_string = ", ".join(ingredients)
            formatted_string = ingredient_string.rsplit(',', 1)
            formatted_string = ' and '.join(formatted_string)
            formatted_string = "<span style='{}'>{}</span>".format(color_style, formatted_string)
            st.write("Your food image may contain: ", formatted_string, unsafe_allow_html=True)
                    
def main():
    st_show()
    
    
if __name__ == '__main__':
    main()