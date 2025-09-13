import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import glob
from PIL import Image
import random

class NEUDETDataset(Dataset):
    """
    NEU-DET Dataset loader for object detection
    Labels are in YOLO format: class_id x_center y_center width height (normalized)
    """
    
    def __init__(self, image_paths, label_paths, img_size=200, augment=False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.img_size = img_size
        self.augment = augment
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = self.label_paths[idx]
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            boxes.append([class_id, x_center, y_center, width, height])
        
        # Resize image
        h, w = image.shape[:2]
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Convert boxes to tensor
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 5), dtype=torch.float32)
        
        return image, boxes

def create_data_splits(image_folder, label_folder, test_split=0.2, random_seed=42):
    """
    Create train/test splits for NEU-DET dataset
    """
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
    
    image_paths = sorted(image_paths)
    
    # Get corresponding label files
    label_paths = []
    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_folder, f"{img_name}.txt")
        label_paths.append(label_path)
    
    # Split into train and test
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(
        image_paths, label_paths, 
        test_size=test_split, 
        random_state=random_seed
    )
    
    print(f"Total images: {len(image_paths)}")
    print(f"Train images: {len(train_imgs)}")
    print(f"Test images: {len(test_imgs)}")
    
    return train_imgs, test_imgs, train_labels, test_labels

def create_data_loaders(train_imgs, train_labels, test_imgs, test_labels, 
                       img_size=200, batch_size=16, num_workers=4):
    """
    Create DataLoader instances for training and testing
    """
    # Create datasets
    train_dataset = NEUDETDataset(train_imgs, train_labels, img_size=img_size, augment=True)
    test_dataset = NEUDETDataset(test_imgs, test_labels, img_size=img_size, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, test_loader

def collate_fn(batch):
    """
    Custom collate function for handling variable number of bounding boxes
    """
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    
    # Pad targets to same length
    max_boxes = max(len(target) for target in targets)
    padded_targets = []
    
    for target in targets:
        if len(target) == 0:
            # Empty target
            padded_target = torch.zeros((max_boxes, 5), dtype=torch.float32)
        else:
            # Pad with zeros
            padding = torch.zeros((max_boxes - len(target), 5), dtype=torch.float32)
            padded_target = torch.cat([target, padding], dim=0)
        padded_targets.append(padded_target)
    
    targets = torch.stack(padded_targets, 0)
    
    return images, targets

def visualize_sample(dataset, idx=0, save_path=None):
    """
    Visualize a sample from the dataset
    """
    image, boxes = dataset[idx]
    
    # Convert image back to numpy for visualization
    img_np = image.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    # Draw bounding boxes
    h, w = img_np.shape[:2]
    for box in boxes:
        if box[0] != 0 or box[1] != 0 or box[2] != 0 or box[3] != 0 or box[4] != 0:
            class_id, x_center, y_center, width, height = box
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_np, f'Class {int(class_id)}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    
    return img_np

if __name__ == "__main__":
    # Test the data loader
    from config import Config
    
    config = Config()
    
    # Create data splits
    train_imgs, test_imgs, train_labels, test_labels = create_data_splits(
        config.IMAGE_FOLDER, 
        config.LABEL_FOLDER, 
        config.TEST_SPLIT, 
        config.RANDOM_SEED
    )
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_imgs, train_labels, test_imgs, test_labels,
        config.IMG_SIZE, config.BATCH_SIZE
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a sample
    train_dataset = NEUDETDataset(train_imgs, train_labels, config.IMG_SIZE)
    sample_img = visualize_sample(train_dataset, 0, "sample_visualization.jpg")
    print("Sample visualization saved as 'sample_visualization.jpg'")
