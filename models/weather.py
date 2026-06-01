import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

CSV_PATH = "data/weather.csv"
MODEL_PATH = "models/weather_model.pkl"
_pipeline = None

def train_and_save_model():
    """Trains a weather summary classifier model on weather.csv and saves it as a pickle file."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset not found at '{CSV_PATH}'")

    print("Loading weather dataset (this may take a few seconds)...")
    df = pd.read_csv(CSV_PATH)
    
    # Strip spaces in headers
    df.columns = df.columns.str.strip()

    # Drop rows with missing targets
    df = df.dropna(subset=['Summary'])

    # Features and target
    features = [
        'Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 
        'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 
        'Pressure (millibars)'
    ]
    X = df[features]
    y = df['Summary']

    # Preprocessing and Pipeline
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(max_depth=12, random_state=42))
    ])

    print("Training Decision Tree Classifier on weather parameters...")
    pipeline.fit(X, y)

    # Save model pipeline
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)

    print("[SUCCESS] Weather model saved successfully!")
    return pipeline

def get_model():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            print("Weather model pickle not found. Training model now...")
            _pipeline = train_and_save_model()
        else:
            with open(MODEL_PATH, 'rb') as f:
                _pipeline = pickle.load(f)
    return _pipeline

def predict_weather(temp, apparent_temp, humidity, wind_speed, wind_bearing, visibility, pressure):
    """
    Predict weather summary (e.g. Partly Cloudy, Foggy, Clear) based on sensor readings.
    """
    pipeline = get_model()

    input_data = pd.DataFrame([{
        'Temperature (C)': float(temp),
        'Apparent Temperature (C)': float(apparent_temp),
        'Humidity': float(humidity),
        'Wind Speed (km/h)': float(wind_speed),
        'Wind Bearing (degrees)': float(wind_bearing),
        'Visibility (km)': float(visibility),
        'Pressure (millibars)': float(pressure)
    }])

    prediction = pipeline.predict(input_data)[0]
    probs = pipeline.predict_proba(input_data)[0]
    confidence = float(max(probs))

    return prediction, confidence

if __name__ == "__main__":
    # Test training and recommendation
    train_and_save_model()
    # Test predict
    summary, conf = predict_weather(9.47, 7.39, 0.89, 14.12, 251.0, 15.83, 1015.13)
    print(f"Test Weather Summary Prediction:")
    print(f"  Predicted Summary: {summary} (Confidence: {conf:.2%})")
