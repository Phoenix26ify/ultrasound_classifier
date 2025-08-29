import os
from PIL import Image
import numpy as np

# Set up paths
base_path = "data"
classes = ["normal", "abnormal"]
splits = ["train", "val"]

# Create folders
for split in splits:
    for cls in classes:
        dir_path = os.path.join(base_path, split, cls)
        os.makedirs(dir_path, exist_ok=True)

        # Generate 5 dummy images per class per split
        for i in range(5):
            # Create a 224x224 RGB image with random pixels
            img_array = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            # Save image
            img.save(os.path.join(dir_path, f"{cls}_{i}.jpg"))

print("✅ Dummy data generated.")
