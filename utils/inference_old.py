# This is pre-processing code for dataset and below are the jobs being done
#Reading images
#Resizing and formatting to model input size
#Normalizing pixel values
#Data augmentation (if training)
#Creating datasets and data loaders
#-- Created by Shreya M ----

import torch
from torchvision import transforms
from PIL import Image
from models.model import build_model
from utils.dataset import efficientnet_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(NUM_CLASSES)
model.load_state_dict(torch.load("best_efficientnet.pth", map_location=device))
model.eval()

def load_model(weights_path, num_classes=2):
    model = build_model(num_classes)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

def predict(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        predicted_class = output.argmax(1).item()
    
    return predicted_class
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = efficientnet_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    return pred  # 0 (normal), 1 (abnormal)

# Example usage
prediction = predict_image("data/images/Kidney-with-infarction-22.24.42.jpg")
print("Predicted class:", "abnormal" if prediction == 1 else "normal")