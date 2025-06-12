# CNU AVSE Deep Learning TERM PROJECT
# 2025 - 1 / Prof. Wonsun Yoo
# 202204305 Seungjae Choi

import os, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize


IMAGE_SIZE   = (200, 300)
BATCH_SIZE   = 32
EPOCHS       = 200
DATASET_PATH = "../dataset/"
CLASS_NAMES  = ["grandeur", "k5", "santafe", "sorento"]
NUM_CLASSES  = len(CLASS_NAMES)
seed = 42
RESULT_DIR = "../result"
os.makedirs(RESULT_DIR, exist_ok=True)


print("\n[INFO] Initializing data generators...")
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.7, 1.3),
    shear_range=0.2,
    validation_split=0.15
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training', shuffle=True, seed=seed
)
val_gen = datagen.flow_from_directory(
    DATASET_PATH, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=False, seed=seed
)


class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))
print("[INFO] Class Weights:", class_weights)


def build_improved_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    # Block1
    x = layers.Conv2D(32, 3, activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    # Block2
    x = layers.Conv2D(64, 3, activation='relu', padding='same',
                    kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    # Global Pooling + Dense
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

model = build_improved_cnn((*IMAGE_SIZE, 3), NUM_CLASSES)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=5e-4),
    metrics=['accuracy']
)
model.summary()


tensorboard_dir = os.path.join("..", "logs", datetime.now().strftime("%Y%m%d_%H%M%S"))

models_dir = os.path.join("..", "models")
os.makedirs(models_dir, exist_ok=True)


callbacks = [
    ModelCheckpoint(os.path.join(models_dir, "best_model.keras"), save_best_only=True, monitor="val_accuracy"),
    EarlyStopping(monitor="val_accuracy", mode="max", patience=30, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_accuracy",mode="max", factor=0.5, patience=10),
    TensorBoard(log_dir=tensorboard_dir)
]


history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    validation_freq=2,
    callbacks=callbacks,
    verbose=2
)


model.save(os.path.join(models_dir, "saved_model_tf.keras"), save_format="tf")


y_true = val_gen.classes
y_prob = model.predict(val_gen)
y_pred = np.argmax(y_prob, axis=1)

# 1. classification report
print("\nClassification Report")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# 2. confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.grid(False)
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
plt.close()

# 3. ROC curve
y_true_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
fpr, tpr, roc_auc = {}, {}, {}
plt.figure(figsize=(10, 8))
for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, label=f"{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(RESULT_DIR, "roc_curve.png"))
plt.close()
