import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import urllib.request
import os

# Model setup
model = models.alexnet(pretrained=False)
model.classifier[6] = nn.Linear(4096, 15)  # change if needed

# Download model from Drive
model_url = "https://drive.google.com/uc?id=1w0R7B9WP9ve_MkournN1pNZPWW2Qr8M4"
model_path = "plant_disease_model.pth"

if not os.path.exists(model_path):
    urllib.request.urlretrieve(model_url, model_path)

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# Dummy class names (replace later)
classes = [f"class_{i}" for i in range(15)]

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# UI
st.title("🌿 Plant Disease Detection")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    img = transform(image).unsqueeze(0)

    outputs = model(img)
    _, predicted = torch.max(outputs, 1)

    st.success(f"Prediction: {classes[predicted.item()]}")