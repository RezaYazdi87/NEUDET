#!/usr/bin/env python3
"""
Demo script showing how to use the NEU-DET object detection system
"""

import os
import sys
import argparse
from config import Config

def print_usage():
    """Print usage instructions"""
    print("""
NEU-DET Object Detection System
===============================

This system implements object detection on the NEU-DET dataset using YOLOv8.

Dataset Requirements:
- 1800 JPG images in 'images/' folder (200x200 pixels)
- 1800 corresponding TXT label files in 'labels/' folder
- Label format: class_id x_center y_center width height (normalized)

Quick Start:
1. Setup: python setup.py
2. Test: python test_setup.py
3. Train: python train.py
4. Quick test: python run_example.py --quick_test

Available Commands:
- setup.py: Install dependencies and check dataset structure
- test_setup.py: Verify installation and test components
- train.py: Train the object detection model
- run_example.py: Run example training with default parameters
- demo.py: Show this help message

Training Parameters:
- --image_folder: Path to image folder (default: images)
- --label_folder: Path to label folder (default: labels)
- --epochs: Number of training epochs (default: 100)
- --batch_size: Batch size for training (default: 16)
- --lr: Learning rate (default: 0.001)
- --test_split: Test split ratio (default: 0.2)

Example Usage:
python train.py --image_folder my_images --label_folder my_labels --epochs 150
python run_example.py --quick_test
python test_setup.py

Output Files:
- outputs/best_model.pt: Trained model
- results/evaluation_report.txt: Detailed metrics
- results/training_curves.png: Training visualization
- results/predictions/: Sample predictions

Evaluation Metrics:
- mAP@0.5: Mean Average Precision at IoU 0.5
- mAP@0.75: Mean Average Precision at IoU 0.75
- mAP@0.5:0.95: Mean Average Precision across IoU thresholds
- Precision, Recall, F1-Score

For more information, see README.md
""")

def check_dataset_structure(image_folder="images", label_folder="labels"):
    """Check if dataset structure is correct"""
    print("Checking dataset structure...")
    
    if not os.path.exists(image_folder):
        print(f"✗ Image folder not found: {image_folder}")
        print("Please create the 'images' folder and add your 1800 JPG images")
        return False
    else:
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"✓ Found {len(image_files)} images in {image_folder}")
    
    if not os.path.exists(label_folder):
        print(f"✗ Label folder not found: {label_folder}")
        print("Please create the 'labels' folder and add your 1800 TXT label files")
        return False
    else:
        label_files = [f for f in os.listdir(label_folder) if f.lower().endswith('.txt')]
        print(f"✓ Found {len(label_files)} labels in {label_folder}")
    
    if len(image_files) != len(label_files):
        print("⚠ Warning: Number of images and labels don't match")
        print(f"  Images: {len(image_files)}, Labels: {len(label_files)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='NEU-DET Object Detection Demo')
    parser.add_argument('--check_dataset', action='store_true',
                       help='Check dataset structure')
    parser.add_argument('--image_folder', type=str, default='images',
                       help='Path to image folder')
    parser.add_argument('--label_folder', type=str, default='labels',
                       help='Path to label folder')
    
    args = parser.parse_args()
    
    if args.check_dataset:
        check_dataset_structure(args.image_folder, args.label_folder)
    else:
        print_usage()

if __name__ == "__main__":
    main()
