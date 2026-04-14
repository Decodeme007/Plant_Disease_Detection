import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load model
model = models.alexnet(pretrained=False)
model.classifier[6] = nn.Linear(4096, 15)  # number of classes

model.load_state_dict(torch.load("plant_disease_model.pth", map_location="cpu"))
model.eval()

# Classes (IMPORTANT - must match dataset)
classes = [
    "Apple___Black_rot", "Apple___healthy", "Corn___Common_rust",
    # add all 38 class names here (or auto load later)
]

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