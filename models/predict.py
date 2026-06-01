import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Class mapping dynamically resolved from directories
DATASET_PATH = "plant_disease"
_model = None
_class_names = None

import json

def get_class_names():
    global _class_names
    if _class_names is None:
        json_path = os.path.join(os.path.dirname(__file__), "disease_classes.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                _class_names = json.load(f)
        else:
            if not os.path.exists(DATASET_PATH):
                raise FileNotFoundError(f"Neither '{json_path}' nor '{DATASET_PATH}' exists.")
            # Get sorted list of directories (matching flow_from_directory ordering)
            _class_names = sorted([
                d for d in os.listdir(DATASET_PATH) 
                if os.path.isdir(os.path.join(DATASET_PATH, d))
            ])
    return _class_names

def load_disease_model():
    global _model
    if _model is None:
        model_path = os.path.join("models", "crop_disease_model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at '{model_path}'. Please run train.py first.")
        # Load the Keras H5 model
        _model = tf.keras.models.load_model(model_path)
    return _model

def clean_label(label):
    if label == "test":
        return "Unknown Test Class"
    if label == "New Plant Diseases Dataset(Augmented)":
        return "Augmented Dataset Root"
    
    # Remove duplicates indicator
    label = label.replace(" copy", "")
    # Replace '___' with ' - ' to separate plant and disease
    label = label.replace("___", " - ")
    # Replace underscores with spaces
    label = label.replace("_", " ")
    # Normalize extra spacing
    label = " ".join(label.split())
    return label

def predict_disease(image_path):
    """
    Predicts the plant disease from a given image file.
    Returns:
        tuple: (cleaned_label, confidence_score, raw_label)
    """
    model = load_disease_model()
    class_names = get_class_names()

    # Load and preprocess image (64x64, matching train.py config)
    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixels
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][pred_idx])

    raw_label = class_names[pred_idx]
    cleaned_label = clean_label(raw_label)

    return cleaned_label, confidence, raw_label

if __name__ == "__main__":
    # Small test on one of the sample test images
    test_img = "plant_disease/test/test/TomatoHealthy1.JPG"
    if os.path.exists(test_img):
        print(f"Testing prediction on: {test_img}")
        label, conf, raw = predict_disease(test_img)
        print(f"Predicted: {label} (Raw: {raw})")
        print(f"Confidence: {conf:.2%}")
    else:
        print("Test image not found. Ensure the directory structure is correct.")
