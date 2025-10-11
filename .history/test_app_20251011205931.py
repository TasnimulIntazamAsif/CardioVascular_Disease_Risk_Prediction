"""
Test script for CVD Risk Assessment Web Application
"""

import requests
import json

def test_web_application():
    """Test the web application endpoints"""
    base_url = "http://localhost:5000"
    
    try:
        # Test main page
        print("Testing main page...")
        response = requests.get(base_url)
        if response.status_code == 200:
            print("✅ Main page loaded successfully")
        else:
            print(f"❌ Main page failed: {response.status_code}")
            return
        
        # Test prediction endpoint with sample data
        print("\nTesting prediction endpoint...")
        sample_data = {
            'Age': '45',
            'Sex': 'M',
            'Weight (kg)': '80',
            'Height (m)': '1.75',
            'Systolic BP': '130',
            'Diastolic BP': '85',
            'Smoking Status': 'N',
            'Diabetes Status': 'N',
            'Physical Activity Level': 'Moderate',
            'Family History of CVD': 'N'
        }
        
        response = requests.post(f"{base_url}/predict", data=sample_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"Risk Level: {result.get('risk_level', 'Unknown')}")
            print(f"Confidence: {result.get('confidence', 'Unknown')}%")
            print(f"Title: {result.get('suggestions', {}).get('title', 'Unknown')}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the web application.")
        print("Make sure the Flask app is running on http://localhost:5000")
    except Exception as e:
        print(f"❌ Error testing application: {e}")

if __name__ == "__main__":
    print("CVD Risk Assessment Web Application Test")
    print("=" * 40)
    test_web_application()
