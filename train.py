import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from models.model import build_model
from utils.dataset import get_dataloaders, efficientnet_transform
#from config import TRAIN_DIR, ANNOTATION_PATH, BATCH_SIZE, NUM_CLASSES, LEARNING_RATE, NUM_EPOCHS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get train and val dataloaders from the XML + images
train_loader, val_loader = get_dataloaders(
    image_dir=TRAIN_DIR,
    xml_path=ANNOTATION_PATH,
    batch_size=BATCH_SIZE,
    val_ratio=0.2,
    transform=efficientnet_transform  # pass your transform here
)

model = build_model(NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    epoch_train_loss = running_loss / len(train_loader.dataset)

    # Validation loop
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = 100 * correct / len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Save best model
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "best_efficientnet.pth")
        print("Best model saved.")
