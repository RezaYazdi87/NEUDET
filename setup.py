#!/usr/bin/env python3
"""
Setup script for NEU-DET Object Detection Project
"""

import os
import sys
import subprocess
import argparse

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    directories = [
        "outputs",
        "results",
        "results/predictions"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_dataset_structure(image_folder, label_folder):
    """Check if dataset structure is correct"""
    print(f"\nChecking dataset structure...")
    
    if not os.path.exists(image_folder):
        print(f"✗ Image folder not found: {image_folder}")
        return False
    else:
        print(f"✓ Image folder found: {image_folder}")
    
    if not os.path.exists(label_folder):
        print(f"✗ Label folder not found: {label_folder}")
        return False
    else:
        print(f"✓ Label folder found: {label_folder}")
    
    # Count files
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    label_files = [f for f in os.listdir(label_folder) if f.lower().endswith('.txt')]
    
    print(f"✓ Found {len(image_files)} image files")
    print(f"✓ Found {len(label_files)} label files")
    
    if len(image_files) != len(label_files):
        print("⚠ Warning: Number of images and labels don't match")
    
    return True

def test_imports():
    """Test if all required modules can be imported"""
    print("\nTesting imports...")
    
    required_modules = [
        'torch',
        'torchvision',
        'ultralytics',
        'cv2',
        'numpy',
        'matplotlib',
        'PIL',
        'sklearn'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            if module == 'cv2':
                import cv2
            elif module == 'PIL':
                from PIL import Image
            elif module == 'sklearn':
                import sklearn
            else:
                __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nFailed to import: {', '.join(failed_imports)}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Setup NEU-DET Object Detection Project')
    parser.add_argument('--image_folder', type=str, default='images',
                       help='Path to image folder')
    parser.add_argument('--label_folder', type=str, default='labels',
                       help='Path to label folder')
    parser.add_argument('--skip_install', action='store_true',
                       help='Skip package installation')
    parser.add_argument('--skip_dataset_check', action='store_true',
                       help='Skip dataset structure check')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NEU-DET Object Detection Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install requirements
    if not args.skip_install:
        if not install_requirements():
            return 1
    
    # Create directories
    create_directories()
    
    # Test imports
    if not test_imports():
        print("\nSome required modules are missing. Please install them manually.")
        return 1
    
    # Check dataset structure
    if not args.skip_dataset_check:
        if not check_dataset_structure(args.image_folder, args.label_folder):
            print("\nDataset structure check failed. Please organize your dataset properly.")
            return 1
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Organize your dataset in the correct folder structure")
    print("2. Run training: python train.py")
    print("3. Check results in the 'outputs' and 'results' folders")
    print("\nFor more information, see README.md")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
