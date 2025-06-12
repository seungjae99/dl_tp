# Car Classification with CNN (Keras)

This project implements a lightweight CNN-based vehicle type classification model using a small custom image dataset (4 classes, ~200 images per class).  
The CNN architecture is designed from scratch (without transfer learning) to achieve high accuracy, even with limited data.

---

## Project Structure
```
dl_tp/
├── dataset/ # Raw or preprocessed image dataset (not included in repo)
├── models/ # Directory for saving best/final models
├── logs/ # TensorBoard training logs
├── src/
│ ├── train.py
│ ├── test.py
│ ├── image_preprocess.py
│ ├── plot.py
│ ├── predict.py
│ ├── shuffle_dataset.py
│ ├── split_dataset.py
│ └── visualize_feature_map.py
└── predict_images/    # Directory for test images
```



---

## Key Features

- Custom-designed, **lightweight CNN** (no transfer learning)
- Conv-BatchNorm-Pool block architecture
- Training callbacks: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`
- TensorBoard logging for monitoring training
- Automatic model saving based on best validation accuracy

---

## Class Categories

- `grandeur`
- `k5`
- `santafe`
- `sorento`

---

## How to Run

### 1. Prepare the dataset
Place your training and validation images in the following structure:

```
data/
├── train/
│   ├── grandeur/
│   ├── k5/
│   ├── santafe/
│   └── sorento/
└── val/
    ├── grandeur/
    ├── k5/
    ├── santafe/
    └── sorento/
```
### 2. Train the model
```
python src/train.py
```
### 3. Run inference on a single image
```
python src/inference.py --img_path predict_images/predict_image.png
```

### Visualize with TensorBoard
```
tensorboard --logdir=logs/
```
Then open `http://localhost:6006` in your browser.

### Saved Models
- Best model: `models/best_model.keras`

- Final model: `models/saved_model_tf.keras`

### Notes
The dataset and trained models are not included in this repository due to file size limitations.

### Development Environment
- Python 3.8+

- TensorFlow / Keras 2.12+

- matplotlib, scikit-learn, opencv-python

🙋‍♂️