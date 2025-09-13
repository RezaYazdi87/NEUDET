import torch
import torch.nn as nn
from ultralytics import YOLO
import os
from config import Config

class NEUDETModel:
    """
    YOLOv8 model wrapper for NEU-DET dataset
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_model(self):
        """
        Create YOLOv8 model for NEU-DET dataset
        """
        # Load pre-trained YOLOv8 model
        self.model = YOLO(self.config.MODEL_NAME)
        
        # Modify the model for single class detection
        # The model will be fine-tuned for defect detection
        print(f"Model loaded: {self.config.MODEL_NAME}")
        print(f"Device: {self.device}")
        
        return self.model
    
    def prepare_data_config(self, train_imgs, train_labels, test_imgs, test_labels):
        """
        Create YAML configuration file for YOLOv8 training
        """
        # Create dataset configuration
        data_config = f"""
# NEU-DET Dataset Configuration
path: {os.path.abspath('.')}  # dataset root dir
train: train_images  # train images (relative to 'path')
val: val_images  # val images (relative to 'path')

# Classes
nc: {self.config.NUM_CLASSES}  # number of classes
names: ['defect']  # class names

# Image settings
img_size: {self.config.IMG_SIZE}
"""
        
        # Save data config
        config_path = os.path.join(self.config.OUTPUT_DIR, 'neudet_config.yaml')
        with open(config_path, 'w') as f:
            f.write(data_config)
        
        # Create symbolic links for YOLOv8 format
        self._create_yolo_dataset_structure(train_imgs, train_labels, test_imgs, test_labels)
        
        return config_path
    
    def _create_yolo_dataset_structure(self, train_imgs, train_labels, test_imgs, test_labels):
        """
        Create YOLOv8 dataset structure with train/val splits
        """
        # Create directories
        train_img_dir = os.path.join(self.config.OUTPUT_DIR, 'train_images')
        train_label_dir = os.path.join(self.config.OUTPUT_DIR, 'train_labels')
        val_img_dir = os.path.join(self.config.OUTPUT_DIR, 'val_images')
        val_label_dir = os.path.join(self.config.OUTPUT_DIR, 'val_labels')
        
        for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Copy training data
        for i, (img_path, label_path) in enumerate(zip(train_imgs, train_labels)):
            img_name = os.path.basename(img_path)
            label_name = os.path.basename(label_path)
            
            # Copy image
            import shutil
            shutil.copy2(img_path, os.path.join(train_img_dir, img_name))
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(train_label_dir, label_name))
        
        # Copy validation data
        for i, (img_path, label_path) in enumerate(zip(test_imgs, test_labels)):
            img_name = os.path.basename(img_path)
            label_name = os.path.basename(label_path)
            
            # Copy image
            shutil.copy2(img_path, os.path.join(val_img_dir, img_name))
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(val_label_dir, label_name))
        
        print(f"Created YOLO dataset structure:")
        print(f"  Train images: {len(train_imgs)}")
        print(f"  Val images: {len(test_imgs)}")
    
    def train(self, data_config_path, epochs=None, batch_size=None, lr=None):
        """
        Train the YOLOv8 model
        """
        if self.model is None:
            self.create_model()
        
        # Training parameters
        train_params = {
            'data': data_config_path,
            'epochs': epochs or self.config.EPOCHS,
            'batch': batch_size or self.config.BATCH_SIZE,
            'imgsz': self.config.IMG_SIZE,
            'lr0': lr or self.config.LEARNING_RATE,
            'weight_decay': self.config.WEIGHT_DECAY,
            'device': self.device,
            'project': self.config.OUTPUT_DIR,
            'name': 'neudet_training',
            'save': True,
            'save_period': 10,
            'patience': 20,
            'conf': self.config.CONFIDENCE_THRESHOLD,
            'iou': self.config.IOU_THRESHOLD,
            'plots': True,
            'val': True,
            'verbose': True
        }
        
        print("Starting training...")
        print(f"Training parameters: {train_params}")
        
        # Train the model
        results = self.model.train(**train_params)
        
        # Save the best model
        best_model_path = os.path.join(self.config.OUTPUT_DIR, 'neudet_training', 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, self.config.MODEL_SAVE_PATH)
            print(f"Best model saved to: {self.config.MODEL_SAVE_PATH}")
        
        return results
    
    def evaluate(self, data_config_path, model_path=None):
        """
        Evaluate the model on test set
        """
        if model_path is None:
            model_path = self.config.MODEL_SAVE_PATH
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load the trained model
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(
            data=data_config_path,
            imgsz=self.config.IMG_SIZE,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            device=self.device,
            plots=True,
            save_json=True
        )
        
        return results
    
    def predict(self, image_path, model_path=None, conf_threshold=None):
        """
        Make prediction on a single image
        """
        if model_path is None:
            model_path = self.config.MODEL_SAVE_PATH
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load the trained model
        model = YOLO(model_path)
        
        # Make prediction
        results = model(
            image_path,
            conf=conf_threshold or self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            imgsz=self.config.IMG_SIZE,
            device=self.device
        )
        
        return results
    
    def get_map_metrics(self, results):
        """
        Extract mAP metrics from validation results
        """
        if hasattr(results, 'box'):
            metrics = results.box
            map50 = metrics.map50  # mAP@0.5
            map75 = metrics.map75  # mAP@0.75
            map = metrics.map      # mAP@0.5:0.95
            precision = metrics.mp  # mean precision
            recall = metrics.mr    # mean recall
            
            return {
                'mAP@0.5': map50,
                'mAP@0.75': map75,
                'mAP@0.5:0.95': map,
                'precision': precision,
                'recall': recall
            }
        else:
            return None

if __name__ == "__main__":
    # Test the model
    config = Config()
    model = NEUDETModel(config)
    
    print("Model created successfully!")
    print(f"Device: {model.device}")
