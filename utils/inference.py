#Single and Batch Inference
import os
import torch
from PIL import Image
from models.model import build_model
from utils.dataset import efficientnet_transform
from config import NUM_CLASSES  # Add this line

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(NUM_CLASSES)
model.load_state_dict(torch.load("best_efficientnet.pth", map_location=device))
model.eval()

def infer_image(img_path):
    image = Image.open(img_path).convert("RGB")
    tensor = efficientnet_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred

def infer_batch(image_folder):
    import os
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Folder not found: {image_folder}")
    files = os.listdir(image_folder)
    if len(files) == 0:
        raise RuntimeError(f"No images found in folder {image_folder}")
    results = {}
    for fname in files:
        if fname.lower().endswith((".jpg", ".png")):
            path = os.path.join(image_folder, fname)
            pred = infer_image(path)
            results[fname] = pred
    return results
