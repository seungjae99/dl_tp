# Predict

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


CLASS_NAMES = ["grandeur", "k5", "santafe", "sorento"]


def preprocess_image(img_path, target_size=(200, 300)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(model_path, img_path):
    # 모델 불러오기
    model = tf.keras.models.load_model(model_path)

    # 이미지 전처리
    img_array = preprocess_image(img_path)

    # 예측
    preds = model.predict(img_array)[0]  # shape: (num_classes,)

    # 결과 출력
    print("* Softmax 확률 (클래스별):")
    for idx, prob in enumerate(preds):
        print(f"  {CLASS_NAMES[idx]:<10}: {prob:.4f}")

    predicted_idx = np.argmax(preds)
    print(f"\n* 최종 예측 클래스: {CLASS_NAMES[predicted_idx]} (index: {predicted_idx})")
    print("")


if __name__ == "__main__":
    img_path = '../predict_images/predict_image6.png'
    model_path = '../models/best_model.keras'
    predict_image(model_path, img_path)
