#!/usr/bin/env python3
"""
Test the updated Flask app with proper feature handling
"""

import requests
import json

def test_updated_flask():
    """Test the updated Flask app"""
    try:
        # Sample test data
        test_data = {
            'Age': 45.0,
            'Sex': 'M',
            'Weight (kg)': 80.0,
            'Height (m)': 1.75,
            'Systolic BP': 130.0,
            'Diastolic BP': 85.0,
            'Smoking Status': 'N',
            'Diabetes Status': 'N',
            'Physical Activity Level': 'Moderate',
            'Family History of CVD': 'N',
            'Height (cm)': 175.0,
            'Abdominal Circumference (cm)': 90.0,
            'Total Cholesterol (mg/dL)': 200.0,
            'HDL (mg/dL)': 50.0,
            'Fasting Blood Sugar (mg/dL)': 100.0,
            'Estimated LDL (mg/dL)': 120.0
        }
        
        print("Testing updated Flask app...")
        print(f"Test data: {test_data}")
        
        # Send request to Flask app
        response = requests.post('http://localhost:5000/predict', json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nStatus Code: {response.status_code}")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            print(f"\nUpdated Flask App Results:")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Confidence: {result['confidence']:.1f}%")
            print(f"Title: {result['suggestions']['title']}")
            
            print(f"\nRecommendations:")
            for i, suggestion in enumerate(result['suggestions']['suggestions'], 1):
                print(f"{i}. {suggestion}")
            
            return True
        else:
            print(f"Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error testing Flask app: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING UPDATED FLASK APP")
    print("=" * 80)
    
    success = test_updated_flask()
    
    if success:
        print("\nUpdated Flask app test completed successfully!")
    else:
        print("\nUpdated Flask app test failed!")
