#!/usr/bin/env python3
"""
Analyze the Flask app's prediction distribution to understand why it's only predicting INTERMEDIARY
"""

import pandas as pd
import numpy as np
import requests
import json

def analyze_prediction_distribution():
    """Analyze what the Flask app is predicting for different risk levels"""
    try:
        df = pd.read_csv('CVD Dataset.csv')
        df = df.dropna(subset=['CVD Risk Level'])
        
        print("Analyzing Flask app prediction distribution...")
        print(f"Dataset distribution:")
        print(df['CVD Risk Level'].value_counts())
        print()
        
        flask_predictions = []
        expected_labels = []
        
        # Test first 50 samples
        for i in range(min(50, len(df))):
            sample_row = df.iloc[i]
            
            # Safe conversion with NaN handling
            def safe_float(value):
                try:
                    if pd.isna(value) or np.isnan(value) or np.isinf(value):
                        return 0.0
                    return float(value)
                except (ValueError, TypeError):
                    return 0.0
            
            def safe_str(value):
                try:
                    if pd.isna(value):
                        return 'N'
                    return str(value)
                except (ValueError, TypeError):
                    return 'N'
            
            flask_data = {
                'Age': safe_float(sample_row['Age']),
                'Sex': safe_str(sample_row['Sex']),
                'Weight (kg)': safe_float(sample_row['Weight (kg)']),
                'Height (m)': safe_float(sample_row['Height (m)']),
                'Systolic BP': safe_float(sample_row['Systolic BP']),
                'Diastolic BP': safe_float(sample_row['Diastolic BP']),
                'Smoking Status': safe_str(sample_row['Smoking Status']),
                'Diabetes Status': safe_str(sample_row['Diabetes Status']),
                'Physical Activity Level': safe_str(sample_row['Physical Activity Level']),
                'Family History of CVD': safe_str(sample_row['Family History of CVD']),
                'Height (cm)': safe_float(sample_row['Height (cm)']),
                'Abdominal Circumference (cm)': safe_float(sample_row['Abdominal Circumference (cm)']),
                'Total Cholesterol (mg/dL)': safe_float(sample_row['Total Cholesterol (mg/dL)']),
                'HDL (mg/dL)': safe_float(sample_row['HDL (mg/dL)']),
                'Fasting Blood Sugar (mg/dL)': safe_float(sample_row['Fasting Blood Sugar (mg/dL)']),
                'Estimated LDL (mg/dL)': safe_float(sample_row['Estimated LDL (mg/dL)'])
            }
            
            expected_risk = sample_row['CVD Risk Level']
            expected_labels.append(expected_risk)
            
            try:
                response = requests.post('http://localhost:5000/predict', json=flask_data)
                
                if response.status_code == 200:
                    result = response.json()
                    flask_risk = result['risk_level']
                    flask_predictions.append(flask_risk)
                else:
                    flask_predictions.append('ERROR')
                    
            except Exception as e:
                flask_predictions.append('ERROR')
        
        # Analyze predictions
        print("Flask app prediction distribution:")
        flask_counts = pd.Series(flask_predictions).value_counts()
        print(flask_counts)
        print()
        
        print("Expected distribution (first 50 samples):")
        expected_counts = pd.Series(expected_labels).value_counts()
        print(expected_counts)
        print()
        
        # Check accuracy by risk level
        print("Accuracy by risk level:")
        for risk_level in ['LOW', 'INTERMEDIARY', 'HIGH']:
            expected_indices = [i for i, label in enumerate(expected_labels) if label == risk_level]
            if expected_indices:
                correct = sum(1 for i in expected_indices if flask_predictions[i] == risk_level)
                total = len(expected_indices)
                accuracy = correct / total * 100
                print(f"{risk_level}: {correct}/{total} = {accuracy:.1f}%")
        
        return flask_predictions, expected_labels
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        return [], []

if __name__ == "__main__":
    print("=" * 80)
    print("ANALYZING FLASK APP PREDICTION DISTRIBUTION")
    print("=" * 80)
    
    flask_preds, expected_labels = analyze_prediction_distribution()
    
    if flask_preds:
        print(f"\nTotal predictions analyzed: {len(flask_preds)}")
        print("Analysis complete!")
    else:
        print("Analysis failed!")
