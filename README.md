# Object Detection with GroundingDINO

This project implements an automated object detection and labeling system using the GroundingDINO model. It processes images and generates YOLO format annotations for training object detection models.

## Project Overview

The system:
- Uses GroundingDINO for zero-shot object detection
- Converts detections to YOLO format
- Organizes detected images and labels into training datasets
- Handles multiple image formats (jpg, png, jpeg)
- Includes memory management for large dataset processing

## Requirements

```python
# Main dependencies
transformers
torch
Pillow
matplotlib
```

## Project Structure

```
project/
│
├── images/                  # Source images organized by class
│   ├── class1/
│   ├── class2/
│   └── ...
│
├── trainImages/            # Processed images and labels
│   ├── class1/
│   │   ├── images/
│   │   └── labels/
│   └── class2/
│       ├── images/
│       └── labels/
│
└── drive/MyDrive/train_images/  # Backup storage location
```

## Key Features

1. **Zero-shot Detection**: Uses GroundingDINO for detecting objects without prior training
2. **YOLO Format Conversion**: Converts bounding boxes to YOLO format
3. **Automated Processing**: Processes entire folders of images
4. **Memory Management**: Includes garbage collection and CUDA memory clearing
5. **Error Handling**: Logs exceptions during processing

## Usage

1. Mount Google Drive (if using Colab):
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Initialize and run the detection:
```python
tm = TrainAutoModel()
folder_list = [i for i in os.listdir("images") if i not in [".ipynb_checkpoints", ".idea", ".venv", ".DS_Store"]]

for folder in folder_list:
    files = tm.list_of_files(folder)
    for file in files:
        detected = tm.model_train(path=str(file), text=f"{folder}.", label_name=folder)
```

## Model Configuration

- Model: `IDEA-Research/grounding-dino-base`
- Image Size: 640x480
- Detection Threshold: 0.35

## Output Format

The system generates YOLO format annotations:
```
<class_id> <x_center> <y_center> <width> <height>
```

## Contributing

Feel free to submit issues and enhancement requests.


## Acknowledgments

- GroundingDINO model by IDEA Research
- YOLO format specifications
