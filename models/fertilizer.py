import os
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

CSV_PATH = "data/fertilizer.csv"
MODEL_PATH = "models/fertilizer_model.pkl"
_pipeline = None

def train_and_save_model():
    """Trains the fertilizer recommendation model and saves it as a pickle file."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset not found at '{CSV_PATH}'")

    # Load data
    df = pd.read_csv(CSV_PATH)
    # Strip any trailing whitespaces in column names (e.g. 'Humidity ')
    df.columns = df.columns.str.strip()

    # Split features and target
    X = df[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
    y = df['Fertilizer Name']

    # Define preprocessor
    categorical_features = ['Soil Type', 'Crop Type']
    numeric_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Define pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train on all available data
    pipeline.fit(X, y)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print("[SUCCESS] Fertilizer model trained and saved successfully!")
    return pipeline

def get_model():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            print("Model pickle not found. Training model now...")
            _pipeline = train_and_save_model()
        else:
            with open(MODEL_PATH, 'rb') as f:
                _pipeline = pickle.load(f)
    return _pipeline

def predict_fertilizer(temp, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous):
    """
    Predict the best fertilizer based on soil and weather parameters.
    """
    pipeline = get_model()

    # Input data formatted as a DataFrame matching columns exactly
    input_data = pd.DataFrame([{
        'Temparature': float(temp),
        'Humidity': float(humidity),
        'Moisture': float(moisture),
        'Soil Type': str(soil_type),
        'Crop Type': str(crop_type),
        'Nitrogen': float(nitrogen),
        'Potassium': float(potassium),
        'Phosphorous': float(phosphorous)
    }])

    prediction = pipeline.predict(input_data)[0]
    probs = pipeline.predict_proba(input_data)[0]
    classes = pipeline.classes_
    confidence = float(max(probs))

    return prediction, confidence

if __name__ == "__main__":
    # Test training and recommendation
    train_and_save_model()
    rec, conf = predict_fertilizer(26, 52, 38, 'Sandy', 'Maize', 37, 0, 0)
    print(f"Test Recommendation: {rec} (Confidence: {conf:.2%})")
