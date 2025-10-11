#!/usr/bin/env python3
"""
Final comprehensive test of Flask app with dataset samples
"""

import pandas as pd
import numpy as np
import requests
import json

def test_flask_with_dataset():
    """Test Flask app with actual dataset samples"""
    try:
        df = pd.read_csv('CVD Dataset.csv')
        df = df.dropna(subset=['CVD Risk Level'])
        
        print("Testing Flask app with actual dataset samples...")
        print(f"Total samples in dataset: {len(df)}")
        
        matches = 0
        total_tested = 0
        
        # Test first 20 samples
        for i in range(min(20, len(df))):
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
            
            try:
                # Test Flask prediction
                response = requests.post('http://localhost:5000/predict', json=flask_data)
                
                if response.status_code == 200:
                    result = response.json()
                    flask_risk = result['risk_level']
                    flask_conf = result['confidence']
                    
                    total_tested += 1
                    
                    if flask_risk == expected_risk:
                        matches += 1
                        status = "MATCH"
                    else:
                        status = "MISMATCH"
                    
                    print(f"Sample {i+1:2d}: Expected={expected_risk:12s} Flask={flask_risk:12s} Conf={flask_conf:5.1f}% {status}")
                    
                else:
                    print(f"Sample {i+1:2d}: Flask error {response.status_code}")
                    
            except Exception as e:
                print(f"Sample {i+1:2d}: Error - {str(e)[:50]}...")
        
        print(f"\n" + "="*60)
        print("FINAL RESULTS:")
        print("="*60)
        print(f"Total samples tested: {total_tested}")
        print(f"Successful matches: {matches}")
        print(f"Match rate: {(matches/total_tested*100):.1f}%" if total_tested > 0 else "No samples tested")
        
        if matches == total_tested and total_tested > 0:
            print("\nSUCCESS: All tested samples match expected results!")
            return True
        else:
            print(f"\nISSUE: {total_tested - matches} samples had mismatches")
            return False
            
    except Exception as e:
        print(f"Error in testing: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("FINAL COMPREHENSIVE TEST - FLASK APP WITH DATASET")
    print("=" * 80)
    
    success = test_flask_with_dataset()
    
    if success:
        print("\nFlask app is working correctly with your dataset!")
    else:
        print("\nFlask app needs further adjustments.")
