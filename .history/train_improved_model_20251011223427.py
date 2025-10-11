#!/usr/bin/env python3
"""
Retrain the Random Forest model with better parameters to fix prediction bias
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

def load_and_preprocess_data():
    """Load and preprocess the CVD dataset with weighted feature fusion"""
    try:
        # Load the dataset
        df = pd.read_csv('CVD Dataset.csv')
        print(f"Dataset loaded: {df.shape}")
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Encode categorical variables
        categorical_features = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD']
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
        
        # Calculate derived features
        df['BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)
        df['Waist-to-Height Ratio'] = df['Abdominal Circumference (cm)'] / df['Height (cm)']
        
        # Handle infinity values
        df['BMI'] = df['BMI'].replace([np.inf, -np.inf], 0.0)
        df['Waist-to-Height Ratio'] = df['Waist-to-Height Ratio'].replace([np.inf, -np.inf], 0.0)
        
        # Define clinical and lifestyle features
        clinical_features = ['Age', 'BMI', 'Abdominal Circumference (cm)', 'Total Cholesterol (mg/dL)', 
                           'HDL (mg/dL)', 'Estimated LDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)', 
                           'Systolic BP', 'Diastolic BP', 'Waist-to-Height Ratio', 'Height (cm)', 
                           'Height (m)', 'Weight (kg)']
        
        lifestyle_features = ['Physical Activity Level', 'Smoking Status', 'Diabetes Status', 'Family History of CVD']
        
        # Prepare clinical data
        clinical_data = df[clinical_features].fillna(0)
        
        # Prepare lifestyle data
        lifestyle_data = df[lifestyle_features].fillna(0)
        
        # Scale features separately
        scaler_clinical = StandardScaler()
        scaler_lifestyle = StandardScaler()
        
        clinical_scaled = scaler_clinical.fit_transform(clinical_data)
        lifestyle_scaled = scaler_lifestyle.fit_transform(lifestyle_data)
        
        # Use optimal weights from previous training
        weight_clinical = 1.252089733851966
        weight_lifestyle = 2.76417517926999
        
        # Apply weighted fusion
        clinical_weighted = clinical_scaled * weight_clinical
        lifestyle_weighted = lifestyle_scaled * weight_lifestyle
        
        # Concatenate weighted features
        X_fused = np.hstack([clinical_weighted, lifestyle_weighted])
        
        print(f"Fused feature matrix shape: {X_fused.shape}")
        
        # Prepare target
        y = df['CVD Risk Level_encoded']
        
        print(f"Target distribution:")
        print(y.value_counts())
        
        return X_fused, y, label_encoders, scaler_clinical, scaler_lifestyle, clinical_features, lifestyle_features
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None, None, None

def train_improved_model():
    """Train an improved Random Forest model"""
    try:
        # Load and preprocess data
        X, y, label_encoders, scaler_clinical, scaler_lifestyle, clinical_features, lifestyle_features = load_and_preprocess_data()
        
        if X is None:
            print("Failed to load data")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Train improved Random Forest model
        # Use parameters that help with class imbalance
        model = RandomForestClassifier(
            n_estimators=200,           # More trees
            max_depth=15,              # Deeper trees
            min_samples_split=5,       # Prevent overfitting
            min_samples_leaf=2,        # Prevent overfitting
            max_features='sqrt',       # Feature selection
            class_weight='balanced',   # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
        
        print("Training improved Random Forest model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['LOW', 'INTERMEDIARY', 'HIGH']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Check prediction distribution
        print(f"\nPrediction distribution on test set:")
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        print(pred_counts)
        
        print(f"\nActual distribution on test set:")
        actual_counts = pd.Series(y_test).value_counts().sort_index()
        print(actual_counts)
        
        # Save improved model and preprocessors
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(model, 'models/rf_improved_model.joblib')
        joblib.dump(scaler_clinical, 'models/scaler_clinical_improved.joblib')
        joblib.dump(scaler_lifestyle, 'models/scaler_lifestyle_improved.joblib')
        
        # Save fusion parameters
        fusion_params = {
            'weight_clinical': 1.252089733851966,
            'weight_lifestyle': 2.76417517926999,
            'clinical_features': clinical_features,
            'lifestyle_features': lifestyle_features
        }
        joblib.dump(fusion_params, 'models/fusion_params_improved.joblib')
        
        # Save label encoders
        for feature, encoder in label_encoders.items():
            joblib.dump(encoder, f'models/{feature}_encoder_improved.joblib')
        
        print(f"\nImproved model saved successfully!")
        print(f"Model accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("TRAINING IMPROVED RANDOM FOREST MODEL")
    print("=" * 80)
    
    success = train_improved_model()
    
    if success:
        print("\nImproved model training completed successfully!")
    else:
        print("\nImproved model training failed!")
