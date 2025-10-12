import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


def parse_cvat_xml(xml_path):
    label_map = {
        "Kidney normal": 0,
        "Kidney abnormal": 1,
        "left kidney abnormal": 1,
        "Right kidney abnormal": 1,
        "Kidney , acute glomerulonephritis": 1,
        "Kidney , urinary obstruction": 1,
        "Kidney , angiomyolipoma": 1,
        "Analgetika-Niere": 1,
        "Niereninfarkt": 1
    }
    fname2label = {}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for image_tag in root.iter("image"):
        fname = image_tag.attrib.get("name")
        assigned_label = None
        for mask_tag in image_tag.iter("mask"):
            label = mask_tag.attrib.get("label")
            if label in label_map:
                assigned_label = label_map[label]
                break
        if assigned_label is None:
            for box_tag in image_tag.iter("box"):
                label = box_tag.attrib.get("label")
                if label in label_map:
                    assigned_label = label_map[label]
                    break
        if assigned_label is not None:
            fname2label[fname] = assigned_label
    return fname2label


class UltrasoundDataset(Dataset):
    def __init__(self, image_dir, fname2label, transform=None):
        self.image_dir = image_dir
        self.fname2label = fname2label
        self.imgs = list(fname2label.keys())
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.image_dir, img_name)
        # fallback if .png not found
        if not os.path.exists(img_path) and img_name.endswith(".png"):
            alt_path = img_path.replace(".png", ".jpg")
            if os.path.exists(alt_path):
                img_path = alt_path
        image = Image.open(img_path).convert("RGB")
        label = self.fname2label[img_name]
        if self.transform:
            image = self.transform(image)
        return image, label


efficientnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def get_dataloaders(image_dir, xml_path, batch_size, val_ratio, transform):
    # Parse label dictionary from XML annotations
    fname2label = parse_cvat_xml(xml_path)
    
    # Get all image filenames from label dict keys
    all_images = list(fname2label.keys())
    
    # Split image filenames into train and validation sets
    train_images, val_images = train_test_split(all_images, test_size=val_ratio, random_state=42)
    
    # Create label dicts for train and validation sets
    train_labels = {fname: fname2label[fname] for fname in train_images}
    val_labels = {fname: fname2label[fname] for fname in val_images}
    
    # Create Dataset objects with correct image folders and labels, passing transforms
    train_dataset = UltrasoundDataset(image_dir, train_labels, transform=transform)
    val_dataset = UltrasoundDataset(image_dir, val_labels, transform=transform)
    
    # Create DataLoaders from datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
