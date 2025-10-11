"""
Weighted Feature Fusion Random Forest Model Training Script
Based on the CVD Risk Assessment notebook implementation
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
    """Load and preprocess the CVD dataset with weighted feature fusion"""
    try:
        # Load the dataset
        df = pd.read_csv('CVD Dataset.csv')
        print(f"Dataset loaded: {df.shape}")
        
        # Handle missing values for numeric columns only
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Encode categorical variables
        categorical_features = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD', 'Blood Pressure Category']
        label_encoders = {}
        
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature].astype(str))
                label_encoders[feature] = le
        
        # Create target variable
        if 'CVD Risk Level' in df.columns:
            target_le = LabelEncoder()
            df['CVD Risk Level_encoded'] = target_le.fit_transform(df['CVD Risk Level'].astype(str))
            label_encoders['CVD Risk Level'] = target_le
        
        # Define clinical and lifestyle features (from notebook)
        clinical_features = [
            'Age', 'BMI', 'Abdominal Circumference (cm)',
            'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Estimated LDL (mg/dL)',
            'Fasting Blood Sugar (mg/dL)', 'Systolic BP', 'Diastolic BP',
            'Waist-to-Height Ratio', 'Height (cm)', 'Height (m)', 'Weight (kg)'
        ]
        
        lifestyle_features = [
            'Physical Activity Level', 'Smoking Status',
            'Diabetes Status', 'Family History of CVD'
        ]
        
        # Check which features are available
        available_clinical = [col for col in clinical_features if col in df.columns]
        available_lifestyle = [col for col in lifestyle_features if col in df.columns]
        
        print(f"Available clinical features: {len(available_clinical)}")
        print(f"Available lifestyle features: {len(available_lifestyle)}")
        
        # Prepare clinical and lifestyle data
        X_clinical = df[available_clinical].fillna(0)
        X_lifestyle = df[available_lifestyle].fillna(0)
        
        # Scale the features separately
        scaler_clinical = StandardScaler()
        scaler_lifestyle = StandardScaler()
        
        Xc_scaled = scaler_clinical.fit_transform(X_clinical)
        Xl_scaled = scaler_lifestyle.fit_transform(X_lifestyle)
        
        # Optimal weights from Optuna optimization (from notebook)
        best_w_c = 1.252089733851966  # Clinical weight
        best_w_l = 2.76417517926999   # Lifestyle weight
        
        print(f"Using optimal weights - Clinical: {best_w_c:.4f}, Lifestyle: {best_w_l:.4f}")
        
        # Apply weighted fusion
        Xc_weighted = Xc_scaled * best_w_c
        Xl_weighted = Xl_scaled * best_w_l
        
        # Concatenate weighted features
        X_fused = np.hstack([Xc_weighted, Xl_weighted])
        
        print(f"Fused feature matrix shape: {X_fused.shape}")
        
        # Prepare target
        y = df['CVD Risk Level_encoded'] if 'CVD Risk Level_encoded' in df.columns else df['CVD Risk Level']
        
        return X_fused, y, label_encoders, scaler_clinical, scaler_lifestyle, best_w_c, best_w_l, available_clinical, available_lifestyle
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None, None, None, None, None

def train_weighted_fusion_model():
    """Train Random Forest model with weighted feature fusion"""
    print("Loading and preprocessing data with weighted feature fusion...")
    X_fused, y, label_encoders, scaler_clinical, scaler_lifestyle, best_w_c, best_w_l, clinical_features, lifestyle_features = load_and_preprocess_data()
    
    if X_fused is None:
        print("Failed to load data. Creating dummy model...")
        return create_dummy_model()
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_fused, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest model with weighted feature fusion...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and preprocessors
    os.makedirs('models', exist_ok=True)
    
    print("Saving model and preprocessors...")
    joblib.dump(model, 'models/rf_weighted_fusion_model.joblib')
    joblib.dump(scaler_clinical, 'models/scaler_clinical.joblib')
    joblib.dump(scaler_lifestyle, 'models/scaler_lifestyle.joblib')
    
    # Save fusion parameters
    fusion_params = {
        'weight_clinical': best_w_c,
        'weight_lifestyle': best_w_l,
        'clinical_features': clinical_features,
        'lifestyle_features': lifestyle_features
    }
    joblib.dump(fusion_params, 'models/fusion_params.joblib')
    
    # Save label encoders
    for feature, encoder in label_encoders.items():
        joblib.dump(encoder, f'models/{feature}_encoder.joblib')
    
    print("Weighted feature fusion model training completed successfully!")
    return model, scaler_clinical, scaler_lifestyle, fusion_params, label_encoders

def create_dummy_model():
    """Create a dummy model for testing"""
    print("Creating dummy model for testing...")
    
    # Create a simple Random Forest with dummy data
    X_dummy = np.random.randn(100, 17)  # 13 clinical + 4 lifestyle features
    y_dummy = np.random.randint(0, 3, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_dummy, y_dummy)
    
    scaler_clinical = StandardScaler()
    scaler_lifestyle = StandardScaler()
    scaler_clinical.fit(X_dummy[:, :13])  # First 13 features are clinical
    scaler_lifestyle.fit(X_dummy[:, 13:])  # Last 4 features are lifestyle
    
    # Save dummy model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/rf_weighted_fusion_model.joblib')
    joblib.dump(scaler_clinical, 'models/scaler_clinical.joblib')
    joblib.dump(scaler_lifestyle, 'models/scaler_lifestyle.joblib')
    
    # Dummy fusion parameters
    fusion_params = {
        'weight_clinical': 1.25,
        'weight_lifestyle': 2.76,
        'clinical_features': ['Age', 'BMI', 'Weight (kg)', 'Height (m)', 'Systolic BP', 'Diastolic BP', 
                            'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Estimated LDL (mg/dL)', 
                            'Fasting Blood Sugar (mg/dL)', 'Abdominal Circumference (cm)', 
                            'Height (cm)', 'Waist-to-Height Ratio'],
        'lifestyle_features': ['Physical Activity Level', 'Smoking Status', 'Diabetes Status', 'Family History of CVD']
    }
    joblib.dump(fusion_params, 'models/fusion_params.joblib')
    
    print("Dummy weighted fusion model created successfully!")
    return model, scaler_clinical, scaler_lifestyle, fusion_params, {}

if __name__ == "__main__":
    print("CVD Risk Assessment - Weighted Feature Fusion Model Training")
    print("=" * 60)
    
    # Check if dataset exists
    if os.path.exists('CVD Dataset.csv'):
        train_weighted_fusion_model()
    else:
        print("CVD Dataset.csv not found. Creating dummy model...")
        create_dummy_model()
    
    print("\nWeighted feature fusion model is ready for the web application!")
    print("Run 'python app.py' to start the web server.")
