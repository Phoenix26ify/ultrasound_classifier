# This is the Custom dataset and preprocessing for dataset and below are the jobs being done
#Model definition (EfficientNet, ResNet, etc.)
#Loss and optimizer setup
#Training loop
#Validation loop
#Saving best model
#(Optional) Inference/prediction function
#-- Created by Shreya M ----

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

def get_dataloaders(train_dir, val_dir, batch_size):
    train_dataset = ImageFolder(train_dir, transform=get_transforms(True))
    val_dataset = ImageFolder(val_dir, transform=get_transforms(False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
