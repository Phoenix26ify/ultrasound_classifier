import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from config import TRAIN_DIR, ANNOTATION_PATH, BATCH_SIZE, NUM_CLASSES
from utils.dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transform, same as training
from torchvision import transforms

efficientnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Get dataloaders (this will split data and apply transforms)
train_loader, val_loader = get_dataloaders(
    image_dir=TRAIN_DIR,
    xml_path=ANNOTATION_PATH,
    batch_size=BATCH_SIZE,
    val_ratio=0.2,
    transform=efficientnet_transform
)

# Load the trained model
from models.model import build_model

model = build_model(NUM_CLASSES)
model.load_state_dict(torch.load("best_efficientnet.pth", map_location=device))
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Print classification report
print(classification_report(all_labels, all_preds, target_names=["normal", "abnormal"]))

# Plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["normal", "abnormal"], yticklabels=["normal", "abnormal"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix")
plt.show()
