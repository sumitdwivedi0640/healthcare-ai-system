import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

IMG_SIZE = 224

# Load once
tumor_model = tf.keras.models.load_model(
    "models/brain_tumor_model_clean.keras")


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = tumor_model.predict(img_array)[0][0]

    if prediction > 0.5:
        return "Tumor Detected", float(prediction)
    else:
        return "No Tumor", float(1 - prediction)

