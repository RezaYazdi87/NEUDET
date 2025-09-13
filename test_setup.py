#!/usr/bin/env python3
"""
Test script to verify NEU-DET setup
"""

import os
import sys
import torch
import numpy as np
from config import Config
from data_loader import create_data_splits, NEUDETDataset
from model import NEUDETModel
from evaluation import NEUDETEvaluator

def test_config():
    """Test configuration"""
    print("Testing configuration...")
    config = Config()
    print(f"✓ Image size: {config.IMG_SIZE}")
    print(f"✓ Batch size: {config.BATCH_SIZE}")
    print(f"✓ Number of classes: {config.NUM_CLASSES}")
    print(f"✓ Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    return True

def test_data_loader():
    """Test data loader with dummy data"""
    print("\nTesting data loader...")
    
    # Create dummy dataset structure
    dummy_images = []
    dummy_labels = []
    
    # Create dummy image files
    os.makedirs("test_images", exist_ok=True)
    os.makedirs("test_labels", exist_ok=True)
    
    for i in range(10):
        # Create dummy image
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        img_path = f"test_images/test_{i:03d}.jpg"
        import cv2
        cv2.imwrite(img_path, img)
        dummy_images.append(img_path)
        
        # Create dummy label
        label_path = f"test_labels/test_{i:03d}.txt"
        with open(label_path, 'w') as f:
            # Random bounding box
            x_center = np.random.uniform(0.2, 0.8)
            y_center = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.1, 0.3)
            height = np.random.uniform(0.1, 0.3)
            f.write(f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")
        dummy_labels.append(label_path)
    
    # Test data splits
    train_imgs, test_imgs, train_labels, test_labels = create_data_splits(
        "test_images", "test_labels", test_split=0.2, random_seed=42
    )
    
    print(f"✓ Created {len(train_imgs)} train images")
    print(f"✓ Created {len(test_imgs)} test images")
    
    # Test dataset
    dataset = NEUDETDataset(train_imgs, train_labels, img_size=200)
    image, boxes = dataset[0]
    
    print(f"✓ Dataset sample - Image shape: {image.shape}")
    print(f"✓ Dataset sample - Boxes shape: {boxes.shape}")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_images")
    shutil.rmtree("test_labels")
    
    return True

def test_model():
    """Test model creation"""
    print("\nTesting model...")
    config = Config()
    model = NEUDETModel(config)
    
    try:
        model.create_model()
        print("✓ Model created successfully")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_evaluator():
    """Test evaluator"""
    print("\nTesting evaluator...")
    config = Config()
    evaluator = NEUDETEvaluator(config)
    print("✓ Evaluator created successfully")
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("NEU-DET Setup Test")
    print("=" * 60)
    
    tests = [
        test_config,
        test_data_loader,
        test_model,
        test_evaluator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("✗ Test failed")
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("✓ All tests passed! Setup is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
