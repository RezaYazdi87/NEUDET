# NEU-DET Object Detection with YOLOv8

This project implements object detection on the NEU-DET dataset using YOLOv8. The dataset contains 1800 images of size 200x200 with corresponding YOLO format labels for defect detection.

## Dataset Format

- **Images**: 1800 JPG files in the `images/` folder
- **Labels**: 1800 corresponding TXT files in the `labels/` folder
- **Label Format**: YOLO format with normalized coordinates
  ```
  class_id x_center y_center width height
  0 0.3775 0.635 0.745 0.36
  ```

## Features

- ✅ YOLOv8 model implementation
- ✅ Random train/test split (80/20 by default)
- ✅ mAP@0.5 evaluation metric
- ✅ Comprehensive training visualization
- ✅ Model evaluation and prediction utilities
- ✅ Support for 200x200 image size
- ✅ Single class detection (defect)

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Organize your dataset:
```
project/
├── images/          # 1800 JPG images
├── labels/          # 1800 TXT label files
├── outputs/         # Training outputs (created automatically)
└── results/         # Evaluation results (created automatically)
```

## Usage

### Basic Training

```bash
python train.py --image_folder images --label_folder labels
```

### Advanced Training with Custom Parameters

```bash
python train.py \
    --image_folder images \
    --label_folder labels \
    --epochs 150 \
    --batch_size 32 \
    --lr 0.001 \
    --test_split 0.2
```

### Test Trained Model

```bash
python -c "from train import test_model; test_model('outputs/best_model.pt', 'path/to/test_image.jpg')"
```

## Configuration

Edit `config.py` to modify:
- Image size (default: 200x200)
- Batch size (default: 16)
- Learning rate (default: 0.001)
- Number of epochs (default: 100)
- Test split ratio (default: 0.2)
- Model architecture (default: YOLOv8n)

## Output Files

After training, the following files will be generated:

- `outputs/best_model.pt` - Best trained model
- `outputs/neudet_training/` - Training logs and checkpoints
- `results/evaluation_report.txt` - Detailed evaluation metrics
- `results/training_curves.png` - Training loss and metric curves
- `results/predictions/` - Sample prediction visualizations
- `results/sample_visualization.jpg` - Sample data visualization

## Evaluation Metrics

The model is evaluated using:
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.75**: Mean Average Precision at IoU threshold 0.75
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds 0.5-0.95
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Model Architecture

This implementation uses YOLOv8 (You Only Look Once version 8) which provides:
- Real-time object detection
- High accuracy on small objects
- Efficient training and inference
- Built-in data augmentation

## Training Process

1. **Data Loading**: Load images and labels from specified folders
2. **Data Splitting**: Random 80/20 train/test split
3. **Data Preprocessing**: Resize images to 200x200, normalize coordinates
4. **Model Training**: Train YOLOv8 with specified parameters
5. **Evaluation**: Calculate mAP@0.5 and other metrics
6. **Visualization**: Generate training curves and prediction samples

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in config.py
2. **Dataset not found**: Ensure image and label folders exist
3. **Label format error**: Verify labels are in YOLO format
4. **Low mAP scores**: Try increasing epochs or adjusting learning rate

### Performance Tips

- Use GPU for faster training
- Increase batch size if you have more GPU memory
- Adjust learning rate based on training progress
- Use data augmentation for better generalization

## License

This project is for educational and research purposes.
