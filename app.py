import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from models.model import build_model
import io

st.title("Ultrasound Image Classification")

# Load model (cache to avoid reload on each interaction)
@st.cache_resource
def load_model():
    NUM_CLASSES = 2
    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load("best_efficientnet.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

efficientnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload an ultrasound image (PNG/JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    input_tensor = efficientnet_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        label = "abnormal" if pred.item() == 1 else "normal"
        confidence = conf.item()
    
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
