import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gdown
import os

# -----------------------------
# 1. MODEL SETUP
# -----------------------------
model = models.alexnet(pretrained=False)
model.classifier[6] = nn.Linear(4096, 15)  # ⚠️ change if your classes differ

# -----------------------------
# 2. DOWNLOAD MODEL FROM DRIVE
# -----------------------------
model_url = "https://drive.google.com/uc?id=1w0R7B9WP9ve_MkournN1pNZPWW2Qr8M4"
model_path = "plant_disease_model.pth"

# Remove corrupted file if exists
if os.path.exists(model_path):
    os.remove(model_path)

# Download model
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Load model
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# -----------------------------
# 3. CLASS NAMES
# -----------------------------
# ⚠️ Replace with your actual class names later
classes = [f"class_{i}" for i in range(15)]

# -----------------------------
# 4. IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# 5. UI
# -----------------------------
st.title("🌿 Plant Disease Detection App")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    outputs = model(img)
    _, predicted = torch.max(outputs, 1)

    st.success(f"Prediction: {classes[predicted.item()]}")