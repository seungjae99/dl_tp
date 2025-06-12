import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE = (200, 300)
BATCH_SIZE = 32
TEST_PATH  = "../dataset/test/"
CLASS_NAMES = ["grandeur", "k5", "santafe", "sorento"]
NUM_CLASSES = len(CLASS_NAMES)


test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


model = tf.keras.models.load_model("../models/best_model.keras")
print("\n* Model Loading COMPLETE\n")


y_true = test_gen.classes
y_prob = model.predict(test_gen)
y_pred = np.argmax(y_prob, axis=1)



TEST_RESULT_DIR = "../test_result"
os.makedirs(TEST_RESULT_DIR, exist_ok=True)

# 1. classification report
print("\nClassification Report")
report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
print(report_str)


with open(os.path.join(TEST_RESULT_DIR, "classification_report.txt"), "w") as f:
    f.write(report_str)

# 2. confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.grid(False)
plt.savefig(os.path.join(TEST_RESULT_DIR, "confusion_matrix.png"))
plt.close()

# 3. ROC curve
y_true_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))

fpr = dict()
tpr = dict()
roc_auc = dict()

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
plt.savefig(os.path.join(TEST_RESULT_DIR, "roc_curve.png"))
plt.close()
