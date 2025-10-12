import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'images')
ANNOTATION_PATH = os.path.join(BASE_DIR, 'data', 'annotations.xml')

BATCH_SIZE = 32
NUM_CLASSES = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
