import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from config import Config
from data_loader import create_data_splits, create_data_loaders, visualize_sample
from model import NEUDETModel
from evaluation import NEUDETEvaluator

def main():
    """
    Main training script for NEU-DET object detection
    """
    parser = argparse.ArgumentParser(description='Train NEU-DET Object Detection Model')
    parser.add_argument('--image_folder', type=str, default='images', 
                       help='Path to image folder')
    parser.add_argument('--label_folder', type=str, default='labels', 
                       help='Path to label folder')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--test_split', type=float, default=0.2, 
                       help='Test split ratio')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    config.IMAGE_FOLDER = args.image_folder
    config.LABEL_FOLDER = args.label_folder
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.TEST_SPLIT = args.test_split
    
    print("=" * 60)
    print("NEU-DET Object Detection Training")
    print("=" * 60)
    print(f"Image folder: {config.IMAGE_FOLDER}")
    print(f"Label folder: {config.LABEL_FOLDER}")
    print(f"Image size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Test split: {config.TEST_SPLIT}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    
    # Check if dataset folders exist
    if not os.path.exists(config.IMAGE_FOLDER):
        raise FileNotFoundError(f"Image folder not found: {config.IMAGE_FOLDER}")
    if not os.path.exists(config.LABEL_FOLDER):
        raise FileNotFoundError(f"Label folder not found: {config.LABEL_FOLDER}")
    
    # Create data splits
    print("\n1. Creating data splits...")
    train_imgs, test_imgs, train_labels, test_labels = create_data_splits(
        config.IMAGE_FOLDER, 
        config.LABEL_FOLDER, 
        config.TEST_SPLIT, 
        config.RANDOM_SEED
    )
    
    # Create data loaders
    print("\n2. Creating data loaders...")
    train_loader, test_loader = create_data_loaders(
        train_imgs, train_labels, test_imgs, test_labels,
        config.IMG_SIZE, config.BATCH_SIZE
    )
    
    # Visualize sample data
    print("\n3. Visualizing sample data...")
    from data_loader import NEUDETDataset
    train_dataset = NEUDETDataset(train_imgs, train_labels, config.IMG_SIZE)
    sample_img = visualize_sample(train_dataset, 0, 
                                os.path.join(config.RESULTS_DIR, "sample_visualization.jpg"))
    
    # Initialize model
    print("\n4. Initializing model...")
    model = NEUDETModel(config)
    model.create_model()
    
    # Prepare data configuration for YOLOv8
    print("\n5. Preparing data configuration...")
    data_config_path = model.prepare_data_config(train_imgs, train_labels, test_imgs, test_labels)
    
    # Train model
    print("\n6. Starting training...")
    start_time = datetime.now()
    
    try:
        results = model.train(
            data_config_path, 
            epochs=config.EPOCHS, 
            batch_size=config.BATCH_SIZE, 
            lr=config.LEARNING_RATE
        )
        
        training_time = datetime.now() - start_time
        print(f"\nTraining completed in: {training_time}")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return
    
    # Evaluate model
    print("\n7. Evaluating model...")
    evaluator = NEUDETEvaluator(config)
    
    try:
        # Calculate detailed metrics
        metrics = evaluator.calculate_detailed_metrics(
            config.MODEL_SAVE_PATH, 
            test_loader
        )
        
        # Generate evaluation report
        report_path = os.path.join(config.RESULTS_DIR, "evaluation_report.txt")
        evaluator.generate_evaluation_report(
            config.MODEL_SAVE_PATH, 
            test_loader, 
            report_path
        )
        
        # Plot training curves
        training_dir = os.path.join(config.OUTPUT_DIR, 'neudet_training')
        curves_path = os.path.join(config.RESULTS_DIR, "training_curves.png")
        evaluator.plot_training_curves(training_dir, curves_path)
        
        # Plot confusion matrix
        confusion_matrix_path = evaluator.plot_confusion_matrix(config.MODEL_SAVE_PATH)
        
        # Visualize predictions on test set
        print("\n8. Visualizing predictions...")
        pred_dir = os.path.join(config.RESULTS_DIR, "predictions")
        evaluator.visualize_predictions(
            config.MODEL_SAVE_PATH, 
            test_imgs[:10], 
            test_labels[:10], 
            pred_dir
        )
        
        # Print final results
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Training time: {training_time}")
        print(f"Final mAP@0.5: {metrics['mAP@0.5']:.4f}")
        print(f"Final mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
        print(f"Final Precision: {metrics['precision']:.4f}")
        print(f"Final Recall: {metrics['recall']:.4f}")
        print(f"Final F1-Score: {metrics['f1_score']:.4f}")
        print(f"\nModel saved to: {config.MODEL_SAVE_PATH}")
        print(f"Results saved to: {config.RESULTS_DIR}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        print("Training completed but evaluation failed.")
    
    print("\nTraining script completed!")

def test_model(model_path, image_path):
    """
    Test the trained model on a single image
    """
    config = Config()
    model = NEUDETModel(config)
    
    print(f"Testing model on: {image_path}")
    results = model.predict(image_path, model_path)
    
    # Print results
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            print(f"Found {len(boxes)} detections:")
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                print(f"  Detection {i+1}: bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), conf={conf:.3f}")
        else:
            print("No detections found")

if __name__ == "__main__":
    main()
