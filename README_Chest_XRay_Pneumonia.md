# 🫁 Chest X-Ray Pneumonia Detection

A deep learning project for detecting **Pneumonia from Chest X-Ray images** using Convolutional Neural Networks and Transfer Learning with **MobileNetV2**.

The project includes data exploration, image preprocessing, data augmentation, transfer learning, model training, evaluation, and a simple **Gradio web interface** for making predictions on new X-Ray images.

---

## 📌 Project Overview

The goal of this project is to build an image classification model that can classify chest X-ray images into two classes:

- **NORMAL**
- **PNEUMONIA**

The project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle and applies Transfer Learning using a pre-trained MobileNetV2 model.

> ⚠️ This project is for educational/research purposes only and is not intended to replace professional medical diagnosis.

---

## 📂 Dataset

**Dataset:** Chest X-Ray Images (Pneumonia)

**Source:** Kaggle  
**Dataset:** `paultimothymooney/chest-xray-pneumonia`

The dataset contains the following structure:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The notebook creates the training and validation generators from the training directory using an **80/20 validation split**, while the test set is kept for final evaluation.

---

## 🔧 Technologies & Libraries

- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- Gradio
- KaggleHub

---

## 🧹 Data Preprocessing

The X-Ray images are resized to:

```text
150 × 150
```

Pixel values are normalized using:

```python
rescale=1./255
```

### Data Augmentation

The training data uses:

- Rotation: `15°`
- Width shift: `0.1`
- Height shift: `0.1`
- Zoom: `0.15`
- Horizontal flipping
- Nearest-neighbor fill mode

A validation split of **20%** is created from the training data.

The test data is only rescaled and is not augmented.

---

## 🧠 Model Architecture

The project uses **MobileNetV2 pre-trained on ImageNet** as the feature extraction backbone.

The original classification head is removed:

```python
MobileNetV2(
    weights='imagenet',
    include_top=False
)
```

The added classification layers are:

```text
MobileNetV2
     ↓
Global Average Pooling
     ↓
Dense(128, ReLU)
     ↓
Dropout(0.5)
     ↓
Dense(1, Sigmoid)
```

The MobileNetV2 base layers are initially frozen:

```python
base_model.trainable = False
```

### Compilation

```text
Loss       : Binary Crossentropy
Optimizer  : Adam
Learning Rate: 0.0001
Metric     : Accuracy
```

---

## 🏋️ Training

The model is trained for up to:

```text
15 epochs
```

Two callbacks are used:

### Early Stopping

Monitors validation loss and restores the best model weights:

```python
EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)
```

### Reduce Learning Rate

Reduces the learning rate when validation loss stops improving:

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7
)
```

---

## 📊 Model Evaluation

The notebook evaluates the model using:

- Validation Loss
- Validation Accuracy
- Test Loss
- Test Accuracy
- Classification Report
- Confusion Matrix

It also generates:

- Training vs. Validation Accuracy plot
- Training vs. Validation Loss plot
- Test-set Confusion Matrix

The classification report includes:

- Precision
- Recall
- F1-score
- Support

### Results

Run the notebook to generate the latest evaluation metrics. The exact final test metrics are intentionally not hard-coded here because the notebook calculates them during execution.

---

## 💾 Saved Model

After training, the model is saved as:

```text
improved_cnn_model.h5
```

---

## 🌐 Gradio Interface

The project includes a simple Gradio interface that allows a user to upload a chest X-ray image and receive a prediction.

### Prediction Classes

```text
NORMAL
PNEUMONIA
```

The interface also displays the model's confidence score.

The image is resized to:

```text
150 × 150
```

and normalized before being passed to the model.

### Example Workflow

```text
Upload X-Ray
     ↓
Resize Image
     ↓
Normalize Pixels
     ↓
MobileNetV2 Model
     ↓
Prediction
     ↓
NORMAL / PNEUMONIA
```

The notebook launches Gradio with:

```python
interface.launch(share=True, debug=True)
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ziaddsaid90-prog/<repository-name>.git
cd <repository-name>
```

### 2. Install the required libraries

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn gradio kagglehub
```

### 3. Open the notebook

Open:

```text
Untitled17 (1).ipynb
```

using Jupyter Notebook, JupyterLab, Google Colab, or Kaggle.

### 4. Run the notebook

Run the cells in order:

1. Download dataset
2. Explore dataset structure
3. Count images
4. Display sample X-Rays
5. Create image generators
6. Build MobileNetV2 transfer-learning model
7. Train the model
8. Evaluate validation performance
9. Plot training history
10. Evaluate on the test set
11. Generate classification report and confusion matrix
12. Save the trained model
13. Launch the Gradio interface

---

## 📁 Project Structure

```text
Chest-XRay-Pneumonia-Detection/
│
├── Untitled17 (1).ipynb
├── improved_cnn_model.h5
├── README.md
└── screenshots/
    ├── sample-xrays.png
    ├── training-accuracy.png
    ├── training-loss.png
    └── confusion-matrix.png
```

> The `screenshots/` folder is optional and can be added later to make the GitHub repository more visual.

---

## 🔍 Key Features

- Chest X-Ray image classification
- Binary classification: NORMAL vs PNEUMONIA
- Transfer Learning with MobileNetV2
- Image augmentation
- Early stopping
- Adaptive learning rate reduction
- Classification report
- Confusion matrix
- Training/validation performance visualization
- Saved trained model
- Interactive Gradio prediction interface

---

## 🎯 Learning Objectives

This project demonstrates practical experience with:

- Computer Vision
- Deep Learning
- Convolutional Neural Networks
- Transfer Learning
- Image preprocessing
- Data augmentation
- Model evaluation
- TensorFlow/Keras
- Model deployment with Gradio

---

## 👨‍💻 Author

**Ziad Said**

GitHub:  
https://github.com/ziaddsaid90-prog

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes. The model should not be used as a standalone medical diagnostic system. Medical decisions should always be made by qualified healthcare professionals.
