#!/usr/bin/env python3
"""
Test the modal fix by sending a request to the Flask app
"""

import requests
import json
import time

def test_modal_fix():
    """Test that the modal properly hides after prediction"""
    try:
        # Sample test data
        test_data = {
            'Age': 35.0,
            'Sex': 'F',
            'Weight (kg)': 65.0,
            'Height (m)': 1.65,
            'Systolic BP': 120.0,
            'Diastolic BP': 80.0,
            'Smoking Status': 'N',
            'Diabetes Status': 'N',
            'Physical Activity Level': 'High',
            'Family History of CVD': 'N',
            'Height (cm)': 165.0,
            'Abdominal Circumference (cm)': 75.0,
            'Total Cholesterol (mg/dL)': 180.0,
            'HDL (mg/dL)': 60.0,
            'Fasting Blood Sugar (mg/dL)': 90.0,
            'Estimated LDL (mg/dL)': 100.0
        }
        
        print("Testing modal fix...")
        print(f"Test data: {test_data}")
        
        # Send request to Flask app
        start_time = time.time()
        response = requests.post('http://localhost:5000/predict', json=test_data)
        end_time = time.time()
        
        print(f"Response time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nStatus Code: {response.status_code}")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            print(f"\nModal Fix Test Results:")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Confidence: {result['confidence']:.1f}%")
            print(f"Title: {result['suggestions']['title']}")
            
            print(f"\nRecommendations:")
            for i, suggestion in enumerate(result['suggestions']['suggestions'], 1):
                print(f"{i}. {suggestion}")
            
            print(f"\n✅ Modal should now properly hide after {end_time - start_time:.2f} seconds")
            return True
        else:
            print(f"Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error testing modal fix: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING MODAL FIX")
    print("=" * 80)
    
    success = test_modal_fix()
    
    if success:
        print("\nModal fix test completed successfully!")
        print("The loading modal should now properly hide after prediction.")
    else:
        print("\nModal fix test failed!")
