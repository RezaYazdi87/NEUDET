import os

class Config:
    # Dataset paths
    IMAGE_FOLDER = "images"  # Path to folder containing 1800 jpg images
    LABEL_FOLDER = "labels"  # Path to folder containing 1800 txt label files
    
    # Image settings
    IMG_SIZE = 200
    IMG_CHANNELS = 3
    
    # Training settings
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0005
    
    # Train/Test split
    TEST_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # Model settings
    MODEL_NAME = "yolov8n.pt"  # YOLOv8 nano for faster training
    NUM_CLASSES = 1  # NEU-DET has 1 class (defect)
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    
    # Output paths
    OUTPUT_DIR = "outputs"
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
    
    # Evaluation settings
    MAP_IOU_THRESHOLD = 0.5  # mAP@0.5
    
    def __init__(self):
        # Create output directories
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
