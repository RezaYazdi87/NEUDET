import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import os
import cv2
from config import Config

class NEUDETEvaluator:
    """
    Evaluation utilities for NEU-DET object detection model
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def plot_training_curves(self, results_dir, save_path=None):
        """
        Plot training curves from YOLOv8 training results
        """
        results_csv = os.path.join(results_dir, 'results.csv')
        
        if not os.path.exists(results_csv):
            print(f"Results CSV not found: {results_csv}")
            return
        
        # Read results
        import pandas as pd
        df = pd.read_csv(results_csv)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('NEU-DET Training Results', fontsize=16)
        
        # Plot loss curves
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue')
        axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='red')
        axes[0, 0].set_title('Box Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot mAP curves
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='green')
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='orange')
        axes[0, 1].set_title('mAP Metrics')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision and recall
        axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='purple')
        axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='brown')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot learning rate
        axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='Learning Rate', color='red')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        
        plt.show()
    
    def visualize_predictions(self, model_path, image_paths, label_paths=None, 
                            save_dir=None, max_images=10):
        """
        Visualize model predictions on sample images
        """
        # Load model
        model = YOLO(model_path)
        
        # Create save directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Process images
        for i, img_path in enumerate(image_paths[:max_images]):
            if not os.path.exists(img_path):
                continue
            
            # Load image
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            # Make prediction
            results = model(img_path, conf=self.config.CONFIDENCE_THRESHOLD)
            
            # Draw predictions
            pred_img = image_rgb.copy()
            
            # Draw ground truth if available
            if label_paths and i < len(label_paths) and os.path.exists(label_paths[i]):
                with open(label_paths[i], 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) == 5:
                                class_id, x_center, y_center, width, height = map(float, parts)
                                x1 = int((x_center - width/2) * w)
                                y1 = int((y_center - height/2) * h)
                                x2 = int((x_center + width/2) * w)
                                y2 = int((y_center + height/2) * h)
                                
                                # Draw ground truth in green
                                cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(pred_img, 'GT', (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw predictions
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        # Draw prediction in red
                        cv2.rectangle(pred_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(pred_img, f'Pred: {conf:.2f}', (int(x1), int(y1)-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Save or display
            if save_dir:
                save_path = os.path.join(save_dir, f'prediction_{i:03d}.jpg')
                cv2.imwrite(save_path, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
            else:
                plt.figure(figsize=(10, 8))
                plt.imshow(pred_img)
                plt.title(f'Predictions on {os.path.basename(img_path)}')
                plt.axis('off')
                plt.show()
        
        if save_dir:
            print(f"Predictions saved to: {save_dir}")
    
    def calculate_detailed_metrics(self, model_path, test_loader, class_names=['defect']):
        """
        Calculate detailed evaluation metrics
        """
        # Load model
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(
            data=os.path.join(self.config.OUTPUT_DIR, 'neudet_config.yaml'),
            imgsz=self.config.IMG_SIZE,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            device=self.device,
            plots=True,
            save_json=True
        )
        
        # Extract metrics
        metrics = results.box
        
        detailed_metrics = {
            'mAP@0.5': metrics.map50,
            'mAP@0.75': metrics.map75,
            'mAP@0.5:0.95': metrics.map,
            'precision': metrics.mp,
            'recall': metrics.mr,
            'f1_score': 2 * (metrics.mp * metrics.mr) / (metrics.mp + metrics.mr + 1e-16)
        }
        
        # Per-class metrics
        if hasattr(metrics, 'ap50'):
            detailed_metrics['per_class_ap50'] = metrics.ap50.tolist()
        if hasattr(metrics, 'ap'):
            detailed_metrics['per_class_ap'] = metrics.ap.tolist()
        
        return detailed_metrics
    
    def plot_confusion_matrix(self, model_path, save_path=None):
        """
        Plot confusion matrix from validation results
        """
        # Load model
        model = YOLO(model_path)
        
        # Run validation to get confusion matrix
        results = model.val(
            data=os.path.join(self.config.OUTPUT_DIR, 'neudet_config.yaml'),
            imgsz=self.config.IMG_SIZE,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            device=self.device,
            plots=True
        )
        
        # The confusion matrix should be saved in the results directory
        confusion_matrix_path = os.path.join(self.config.OUTPUT_DIR, 'neudet_training', 'confusion_matrix.png')
        
        if os.path.exists(confusion_matrix_path):
            print(f"Confusion matrix saved to: {confusion_matrix_path}")
            return confusion_matrix_path
        else:
            print("Confusion matrix not found in results")
            return None
    
    def generate_evaluation_report(self, model_path, test_loader, save_path=None):
        """
        Generate comprehensive evaluation report
        """
        print("Generating evaluation report...")
        
        # Calculate metrics
        metrics = self.calculate_detailed_metrics(model_path, test_loader)
        
        # Create report
        report = f"""
NEU-DET Object Detection Model Evaluation Report
===============================================

Model: {model_path}
Image Size: {self.config.IMG_SIZE}x{self.config.IMG_SIZE}
Confidence Threshold: {self.config.CONFIDENCE_THRESHOLD}
IoU Threshold: {self.config.IOU_THRESHOLD}

Performance Metrics:
-------------------
mAP@0.5: {metrics['mAP@0.5']:.4f}
mAP@0.75: {metrics['mAP@0.75']:.4f}
mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}
Precision: {metrics['precision']:.4f}
Recall: {metrics['recall']:.4f}
F1-Score: {metrics['f1_score']:.4f}

Per-Class Metrics:
-----------------
"""
        
        if 'per_class_ap50' in metrics:
            for i, ap50 in enumerate(metrics['per_class_ap50']):
                report += f"Class {i} (defect) AP@0.5: {ap50:.4f}\n"
        
        if 'per_class_ap' in metrics:
            for i, ap in enumerate(metrics['per_class_ap']):
                report += f"Class {i} (defect) AP@0.5:0.95: {ap:.4f}\n"
        
        report += f"""
Dataset Information:
-------------------
Total Classes: {self.config.NUM_CLASSES}
Class Names: {['defect']}

Evaluation completed successfully!
"""
        
        print(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Evaluation report saved to: {save_path}")
        
        return metrics

if __name__ == "__main__":
    # Test evaluation utilities
    config = Config()
    evaluator = NEUDETEvaluator(config)
    
    print("Evaluation utilities created successfully!")
