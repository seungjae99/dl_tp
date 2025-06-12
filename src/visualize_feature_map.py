import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import os


model_path = "../models/best_model.keras"
img_path = "../predict_images/predict_image.png"
output_dir = "../feature_maps"
os.makedirs(output_dir, exist_ok=True)


model = tf.keras.models.load_model(model_path)


img = image.load_img(img_path, target_size=(200, 300))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)


layer_outputs = []
layer_names = []
for layer in model.layers:
    if 'conv' in layer.name.lower() or 'Conv' in layer.__class__.__name__:
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)


activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_array)


imgs_per_row = 16


for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    h_size = layer_activation.shape[1]
    w_size = layer_activation.shape[2]
    n_cols = n_features // imgs_per_row
    display_grid = np.zeros((n_cols * h_size, imgs_per_row * w_size))

    for col in range(n_cols):
        for row in range(imgs_per_row):
            idx = col * imgs_per_row + row
            if idx >= n_features:
                break
            channel_image = layer_activation[0, :, :, idx]
            channel_image -= channel_image.mean()
            channel_image /= (channel_image.std() + 1e-5)
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[
                col * h_size : (col + 1) * h_size,
                row * w_size : (row + 1) * w_size
            ] = channel_image


    fig = plt.figure(figsize=(display_grid.shape[1] / 100.0, display_grid.shape[0] / 100.0), dpi=100)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.axis('off')
    save_path = os.path.join(output_dir, f"{layer_name}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
