"""
Simple model training script for CVD Risk Assessment
This script creates a basic Random Forest model for the web application
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

def load_and_preprocess_data():
    """Load and preprocess the CVD dataset"""
    try:
        # Load the dataset
        df = pd.read_csv('CVD Dataset.csv')
        print(f"Dataset loaded: {df.shape}")
        
        # Handle missing values
        df = df.fillna(df.median())
        
        # Encode categorical variables
        categorical_features = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD', 'Blood Pressure Category']
        label_encoders = {}
        
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature])
                label_encoders[feature] = le
        
        # Create target variable
        if 'CVD Risk Level' in df.columns:
            target_le = LabelEncoder()
            df['CVD Risk Level_encoded'] = target_le.fit_transform(df['CVD Risk Level'])
            label_encoders['CVD Risk Level'] = target_le
        
        # Select features for training
        feature_columns = [
            'Age', 'Weight (kg)', 'Height (m)', 'BMI', 'Abdominal Circumference (cm)',
            'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)',
            'Height (cm)', 'Waist-to-Height Ratio', 'Systolic BP', 'Diastolic BP',
            'Estimated LDL (mg/dL)', 'CVD Risk Score', 'Cholesterol_Ratio', 'BP_Ratio',
            'Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level',
            'Family History of CVD', 'Blood Pressure Category'
        ]
        
        # Ensure all features exist
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"Available features: {len(available_features)}")
        
        # Prepare data
        X = df[available_features].fillna(0)
        y = df['CVD Risk Level_encoded'] if 'CVD Risk Level_encoded' in df.columns else df['CVD Risk Level']
        
        return X, y, label_encoders, available_features
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def train_model():
    """Train a Random Forest model"""
    print("Loading and preprocessing data...")
    X, y, label_encoders, feature_columns = load_and_preprocess_data()
    
    if X is None:
        print("Failed to load data. Creating dummy model...")
        return create_dummy_model()
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and preprocessors
    os.makedirs('models', exist_ok=True)
    
    print("Saving model and preprocessors...")
    joblib.dump(model, 'models/rf_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Save label encoders
    for feature, encoder in label_encoders.items():
        joblib.dump(encoder, f'models/{feature}_encoder.joblib')
    
    print("Model training completed successfully!")
    return model, scaler, label_encoders

def create_dummy_model():
    """Create a dummy model for testing"""
    print("Creating dummy model for testing...")
    
    # Create a simple Random Forest with dummy data
    X_dummy = np.random.randn(100, 22)
    y_dummy = np.random.randint(0, 3, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_dummy, y_dummy)
    
    scaler = StandardScaler()
    scaler.fit(X_dummy)
    
    # Save dummy model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/rf_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    print("Dummy model created successfully!")
    return model, scaler, {}

if __name__ == "__main__":
    print("CVD Risk Assessment Model Training")
    print("=" * 40)
    
    # Check if dataset exists
    if os.path.exists('CVD Dataset.csv'):
        train_model()
    else:
        print("CVD Dataset.csv not found. Creating dummy model...")
        create_dummy_model()
    
    print("\nModel is ready for the web application!")
    print("Run 'python app.py' to start the web server.")
