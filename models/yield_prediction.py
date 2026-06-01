import os
import pickle
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

CSV_PATH = "data/crop_yield.csv"
MODEL_PATH = "models/yield_model.pkl"
CATEGORIES_PATH = "models/yield_categories.json"

_pipeline = None
_categories = None

def train_and_save_model():
    """Trains the crop yield prediction model and saves the pipeline and category metadata."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset not found at '{CSV_PATH}'")

    print("Loading crop yield dataset (this may take a few seconds)...")
    df = pd.read_csv(CSV_PATH)
    
    # Strip spaces from column headers and string values
    df.columns = df.columns.str.strip()
    for col in ['State_Name', 'District_Name', 'Season', 'Crop']:
        df[col] = df[col].astype(str).str.strip()

    # Drop missing target values
    df = df.dropna(subset=['Production'])

    # Features and target
    X = df[['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop', 'Area']]
    y = df['Production']

    # Preprocessing
    categorical_features = ['State_Name', 'District_Name', 'Season', 'Crop']
    numeric_features = ['Crop_Year', 'Area']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Use a DecisionTreeRegressor with moderate depth for fast training on 246k rows
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(max_depth=15, random_state=42))
    ])

    print("Training Decision Tree Regressor...")
    pipeline.fit(X, y)

    # Save model pipeline
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)

    # Save unique categories metadata for UI dropdown filtering
    categories = {
        'states': sorted(df['State_Name'].unique().tolist()),
        'seasons': sorted(df['Season'].unique().tolist()),
        'crops': sorted(df['Crop'].unique().tolist()),
        # Map state -> list of its districts
        'state_districts': {
            state: sorted(df[df['State_Name'] == state]['District_Name'].unique().tolist())
            for state in df['State_Name'].unique()
        }
    }
    with open(CATEGORIES_PATH, 'w') as f:
        json.dump(categories, f)

    print("[SUCCESS] Crop yield model and metadata saved successfully!")
    return pipeline, categories

def load_resources():
    global _pipeline, _categories
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(CATEGORIES_PATH):
            print("Yield model resources not found. Training model now...")
            _pipeline, _categories = train_and_save_model()
        else:
            with open(MODEL_PATH, 'rb') as f:
                _pipeline = pickle.load(f)
            with open(CATEGORIES_PATH, 'r') as f:
                _categories = json.load(f)
    return _pipeline, _categories

def get_categories():
    _, categories = load_resources()
    return categories

def predict_yield(state, district, year, season, crop, area):
    """
    Predict crop production and yield (production / area) based on agricultural features.
    """
    pipeline, _ = load_resources()

    input_data = pd.DataFrame([{
        'State_Name': str(state).strip(),
        'District_Name': str(district).strip(),
        'Crop_Year': int(year),
        'Season': str(season).strip(),
        'Crop': str(crop).strip(),
        'Area': float(area)
    }])

    predicted_production = float(pipeline.predict(input_data)[0])
    # Avoid division by zero
    predicted_yield = predicted_production / max(float(area), 0.001)

    return predicted_production, predicted_yield

if __name__ == "__main__":
    # Test training and recommendation
    pipeline, cats = train_and_save_model()
    # Test predict
    prod, yld = predict_yield('Andhra Pradesh', 'ANANTAPUR', 2000, 'Kharif', 'Rice', 100.0)
    print(f"Test Prediction for 100 hectares of Rice:")
    print(f"  Predicted Production: {prod:.2f} tons")
    print(f"  Projected Yield: {yld:.2f} tons/hectare")
