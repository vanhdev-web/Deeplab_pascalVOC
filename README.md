# Deeplab_pascalVOC

## Description

This repository implements semantic segmentation using the DeeplabV3 model with a MobileNetV3 backbone on the Pascal VOC dataset.  The project allows for both training a Deeplab model from scratch and performing inference on images using a pre-trained model.

## Features and Functionality

*   **Semantic Segmentation:**  Performs pixel-level classification to assign each pixel in an image to a specific object class.
*   **DeeplabV3 with MobileNetV3 Backbone:** Utilizes a lightweight and efficient architecture for semantic segmentation.
*   **Pascal VOC Dataset:**  Specifically designed to work with the Pascal VOC dataset for training and evaluation.
*   **Training:**  Includes a training script (`train_deeplab.py`) to train the Deeplab model on the Pascal VOC dataset.
*   **Inference:**  Provides an inference script (`inference.py`) to perform semantic segmentation on new images.
*   **Visualization:** The inference script generates segmented images, including an overlay on the original image.
*   **TensorBoard Logging:**  The training script logs training progress, loss, accuracy, and Jaccard index to TensorBoard for monitoring.
*   **Checkpoint Saving:** Saves the best performing model during training as `trained_models/best.pt` and the last trained model as `trained_models/last.pt`.
*   **Customizable Training Parameters:** Offers command-line arguments to customize training parameters such as batch size, image size, and number of epochs.

## Technology Stack

*   **Python:**  Primary programming language.
*   **PyTorch:**  Deep learning framework.
*   **Torchvision:**  Provides datasets, model architectures, and image transformations.
*   **NumPy:**  Numerical computation library.
*   **OpenCV (cv2):**  Computer vision library for image processing.
*   **TorchMetrics:**  For calculating accuracy and Jaccard index.
*   **TensorBoard:** For visualizing training progress.
*   **Tqdm:** For displaying progress bars during training and evaluation.

## Prerequisites

Before running the code, ensure you have the following installed:

*   **Python:** (>=3.6)
*   **PyTorch:** (>=1.10) with CUDA support if you want to utilize GPU.
*   **Torchvision:** (>=0.11)
*   **NumPy:**
*   **OpenCV:**
*   **TorchMetrics:**
*   **TensorBoard:**
*   **Tqdm:**

You can install the required packages using pip:

```bash
pip install torch torchvision numpy opencv-python torchmetrics tensorboard tqdm
```

## Installation Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/vanhdev-web/Deeplab_pascalVOC.git
    cd Deeplab_pascalVOC
    ```

2.  **Download the Pascal VOC dataset:**

    The training script expects the dataset to be located at the path specified by the `--data_path` argument. By default, it is set to `C:\Users\dovie\OneDrive\Desktop\vietanh\deeplearning\Pascal_VOC\dataset\archive`.  You can download the Pascal VOC 2012 dataset and organize it accordingly. The script expects the VOC dataset structure after extracting the archive.  If you place the dataset in a different location, update the `--data_path` argument in the training script.

## Usage Guide

### 1. Training the Deeplab Model

To train the Deeplab model, run the `train_deeplab.py` script with the desired arguments.

```bash
python train_deeplab.py --data_path /path/to/pascal/voc/dataset --image_size 224 --epoch 100 --batch_size 4 --logging_folder tensorboard_logs --trained_models trained_models
```

**Arguments:**

*   `--data_path` or `-d`: Path to the Pascal VOC dataset. (Default: `C:\Users\dovie\OneDrive\Desktop\vietanh\deeplearning\Pascal_VOC\dataset\archive`)
*   `--image_size` or `-imgsz`:  Size to which the images will be resized during training. (Default: 224)
*   `--epoch` or `-e`: Number of training epochs. (Default: 100)
*   `--batch_size` or `-b`:  Batch size for training. (Default: 4)
*   `--logging_folder` or `-l`:  Directory to store TensorBoard logs. (Default: `tensorboard`)
*   `--trained_models` or `-t`: Directory to save trained models. (Default: `trained_models`)
*   `--checkpoint_folder` or `-c`: Path to a checkpoint file (`.pt`) to resume training from. If not provided, training starts from scratch. (Default: `None`)

**Monitoring Training Progress:**

You can monitor the training progress using TensorBoard:

```bash
tensorboard --logdir tensorboard_logs
```

### 2. Performing Inference

To perform inference on an image using a trained model, run the `inference.py` script.

```bash
python inference.py --image_path demo/deeplab1.png --checkpoint trained_models/best.pt
```

**Arguments:**

*   `--image_path` or `-i`: Path to the input image. (Default: `demo/deeplab1.png`)
*   `--checkpoint` or `-c`: Path to the trained model checkpoint file (`.pt`). (Default: `trained_models/best.pt`)

The script will generate two output images:

*   `demo/deeplab1_prediction.png`: The segmentation mask with color-coded classes.
*   `demo/deeplab1_overlay.png`: The segmentation mask overlaid on the original image.

## API Documentation

This project does not expose an explicit API.  The core functionality is accessed through the provided scripts and their command-line arguments.

## Contributing Guidelines

Contributions are welcome! To contribute to this project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes.
4.  Test your changes thoroughly.
5.  Submit a pull request with a clear description of your changes.

## License Information

No license is specified for this repository.  All rights are reserved by the owner, unless otherwise stated.

## Contact/Support Information

For questions or support, please contact the repository owner through GitHub issues or by email at the address associated with the GitHub profile.