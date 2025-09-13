#!/usr/bin/env python3
"""
Example script to run NEU-DET object detection training
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run NEU-DET Object Detection Example')
    parser.add_argument('--image_folder', type=str, default='images',
                       help='Path to image folder')
    parser.add_argument('--label_folder', type=str, default='labels',
                       help='Path to label folder')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with minimal epochs')
    
    args = parser.parse_args()
    
    # Adjust parameters for quick test
    if args.quick_test:
        args.epochs = 5
        args.batch_size = 8
        print("Running quick test with 5 epochs...")
    
    print("=" * 60)
    print("NEU-DET Object Detection - Example Run")
    print("=" * 60)
    print(f"Image folder: {args.image_folder}")
    print(f"Label folder: {args.label_folder}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(args.image_folder):
        print(f"Error: Image folder not found: {args.image_folder}")
        print("Please organize your dataset with images in the 'images' folder")
        return 1
    
    if not os.path.exists(args.label_folder):
        print(f"Error: Label folder not found: {args.label_folder}")
        print("Please organize your dataset with labels in the 'labels' folder")
        return 1
    
    # Run training
    cmd = [
        sys.executable, "train.py",
        "--image_folder", args.image_folder,
        "--label_folder", args.label_folder,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size)
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}")
    print("\nStarting training...")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())
