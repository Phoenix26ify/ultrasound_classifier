# This is the main classification code for dataset and below are the jobs being done
#Model definition (EfficientNet, ResNet, etc.)
#Loss and optimizer setup
#Training loop
#Validation loop
#Saving best model
#(Optional) Inference/prediction function
#-- Created by Shreya M ----

import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from models.model import build_model
from utils.dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = get_dataloaders(TRAIN_DIR, VAL_DIR, BATCH_SIZE)
model = build_model(NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss:.4f}")

    torch.save(model.state_dict(), f"efficientnet_epoch{epoch+1}.pth")
