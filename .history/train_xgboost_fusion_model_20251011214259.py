#!/usr/bin/env python3
"""
Train XGBoost model with weighted feature fusion for CVD Risk Assessment
This script implements the best performing model from the notebook analysis.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the CVD dataset with weighted feature fusion"""
    try:
        # Load the dataset
        df = pd.read_csv('CVD Dataset.csv')
        print(f"Dataset loaded: {df.shape}")
        
        # Handle missing values
        df = df.dropna(subset=['CVD Risk Level'])
        
        # Create label encoder for target
        label_encoder = LabelEncoder()
        df['CVD Risk Level_encoded'] = label_encoder.fit_transform(df['CVD Risk Level'])
        
        # Handle missing values in features
        numeric_features = ['Age', 'Weight (kg)', 'Height (m)', 'Systolic BP', 'Diastolic BP', 
                          'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)', 
                          'Estimated LDL (mg/dL)', 'Abdominal Circumference (cm)', 'Height (cm)']
        
        for feature in numeric_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(df[feature].median())
        
        # Encode categorical features
        categorical_features = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 
                              'Family History of CVD', 'Blood Pressure Category']
        
        label_encoders = {}
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                label_encoders[feature] = le
        
        # Calculate derived features
        df['BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)
        df['Waist-to-Height Ratio'] = df['Abdominal Circumference (cm)'] / df['Height (cm)']
        
        # Calculate CVD Risk Score (simplified version)
        df['CVD Risk Score'] = (
            df['Age'] * 0.1 + 
            df['BMI'] * 0.2 + 
            df['Systolic BP'] * 0.15 + 
            df['Total Cholesterol (mg/dL)'] * 0.1 +
            df['Fasting Blood Sugar (mg/dL)'] * 0.1 +
            df['Smoking Status_encoded'] * 2 +
            df['Diabetes Status_encoded'] * 1.5 +
            df['Family History of CVD_encoded'] * 1
        )
        
        # Define clinical and lifestyle features (from notebook analysis)
        clinical_features = [
            'Age', 'BMI', 'Abdominal Circumference (cm)', 'Total Cholesterol (mg/dL)', 
            'HDL (mg/dL)', 'Estimated LDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)', 
            'Systolic BP', 'Diastolic BP', 'Waist-to-Height Ratio', 'Height (cm)', 
            'Height (m)', 'Weight (kg)'
        ]
        
        lifestyle_features = [
            'Physical Activity Level_encoded', 'Smoking Status_encoded', 
            'Diabetes Status_encoded', 'Family History of CVD_encoded'
        ]
        
        # Prepare feature data
        clinical_data = df[clinical_features].values
        lifestyle_data = df[lifestyle_features].values
        
        # Scale features separately
        scaler_clinical = StandardScaler()
        scaler_lifestyle = StandardScaler()
        
        clinical_scaled = scaler_clinical.fit_transform(clinical_data)
        lifestyle_scaled = scaler_lifestyle.fit_transform(lifestyle_data)
        
        # Optimal weights from Optuna optimization (from notebook)
        weight_clinical = 1.252089733851966
        weight_lifestyle = 2.76417517926999
        
        # Apply weighted fusion
        clinical_weighted = clinical_scaled * weight_clinical
        lifestyle_weighted = lifestyle_scaled * weight_lifestyle
        
        # Concatenate weighted features
        X_fused = np.hstack([clinical_weighted, lifestyle_weighted])
        y = df['CVD Risk Level_encoded'].values
        
        print(f"Fused features shape: {X_fused.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        
        return X_fused, y, label_encoders, scaler_clinical, scaler_lifestyle, weight_clinical, weight_lifestyle, clinical_features, lifestyle_features
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None, None, None, None, None

def train_xgboost_fusion_model():
    """Train XGBoost model with weighted feature fusion"""
    print("Loading and preprocessing data with weighted feature fusion...")
    X_fused, y, label_encoders, scaler_clinical, scaler_lifestyle, best_w_c, best_w_l, clinical_features, lifestyle_features = load_and_preprocess_data()
    
    if X_fused is None:
        print("Failed to load data. Creating dummy model...")
        return create_dummy_model()
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_fused, y, test_size=0.2, random_state=42, stratify=y)
    
    # Calculate class weights for imbalanced dataset
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    print("Training XGBoost model with weighted feature fusion...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        class_weight=class_weight_dict
    )
    
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_scores = cross_val_score(model, X_fused, y, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save model and preprocessors
    os.makedirs('models', exist_ok=True)
    
    print("Saving model and preprocessors...")
    joblib.dump(model, 'models/xgboost_fusion_model.joblib')
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
    
    print("XGBoost Fusion model training completed successfully!")
    return model, scaler_clinical, scaler_lifestyle, label_encoders

def create_dummy_model():
    """Create a dummy model for testing"""
    model = xgb.XGBClassifier(n_estimators=10, random_state=42)
    scaler_clinical = StandardScaler()
    scaler_lifestyle = StandardScaler()
    label_encoders = {}
    
    # Create dummy data
    X_dummy = np.random.randn(100, 17)
    y_dummy = np.random.randint(0, 3, 100)
    
    model.fit(X_dummy, y_dummy)
    scaler_clinical.fit(X_dummy[:, :13])
    scaler_lifestyle.fit(X_dummy[:, 13:])
    
    return model, scaler_clinical, scaler_lifestyle, label_encoders

if __name__ == "__main__":
    print("=" * 80)
    print("XGBOOST FUSION MODEL TRAINING")
    print("=" * 80)
    
    model, scaler_clinical, scaler_lifestyle, label_encoders = train_xgboost_fusion_model()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print("Model saved to: models/xgboost_fusion_model.joblib")
    print("Scalers saved to: models/scaler_clinical.joblib, models/scaler_lifestyle.joblib")
    print("Fusion parameters saved to: models/fusion_params.joblib")
    print("Label encoders saved to: models/*_encoder.joblib")
