# 🩻 Ultrasound Image Classifier — Normal vs Abnormal

> A PyTorch-based deep learning project that classifies ultrasound images as **normal** or **abnormal** using EfficientNet-B0 with transfer learning.  
> Created by **Shreya Mitra** 🧠

---

## 📁 Project Structure

ultrasound_classifier/
├── data/
│ ├── train/
│ │ ├── normal/
│ │ └── abnormal/
│ └── val/
│ ├── normal/
│ └── abnormal/
├── models/
│ └── model.py # EfficientNet model builder
├── utils/
│ ├── dataset.py # Preprocessing, transforms, dataloaders
│ └── inference.py # Image inference and prediction
├── train.py # Model training script
├── config.py # Hyperparameters and directory config
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## 🚀 Features

- ✅ Transfer learning with EfficientNet-B0
- ✅ Image augmentation (rotation, flip, normalization)
- ✅ Binary classification: **normal** vs **abnormal**
- ✅ Modular code (training, inference, preprocessing)
- ✅ Easily extendable to multi-disease classification

---

## 🛠 Installation

1. Clone the repo:
```bash
- git clone https://github.com/yourusername/ultrasound_classifier.git
- cd ultrasound_classifier

2. Create a virtual environment (optional but recommended):

- python3 -m venv venv
- source venv/bin/activate

3. Install dependencies:
- pip install -r requirements.txt

🖼️ Dataset Format

Organize your data as:

data/
├── train/
│   ├── normal/
│   └── abnormal/
└── val/
    ├── normal/
    └── abnormal/

Each folder should contain .jpg, .jpeg, or .png ultrasound images.

🧠 Training

Run training with:

 - python train.py

Notes - Epochs, learning rate, and batch size can be adjusted in config.py

        Trained weights are saved automatically after each epoch

🧪 Inference
Use the inference.py script to classify new images:

from utils.inference import load_model, predict

model = load_model('efficientnet_epoch10.pth')
result = predict('path/to/ultrasound.jpg', model)

print("Predicted class:", "Normal" if result == 0 else "Abnormal")


Future Improvements

Add validation accuracy tracking and loss plots

Extend to multi-disease classification (Stage 2)

Visualize predictions with Grad-CAM

Add Streamlit/Gradio web UI for demo

Requirements
torch
torchvision
Pillow
matplotlib

Install via:
pip install -r requirements.txt


📄 License

This project is open-source and MIT-licensed.

👩‍💻 Author

Built and maintained by Shreya Mitra
🔗 LinkedIn
 | ✉️ shreya.mitra@email.com
