import requests
import json

# Test the prediction endpoint
url = "http://localhost:5000/predict"

# Sample data
data = {
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

try:
    response = requests.post(url, data=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Risk Level: {result.get('risk_level')}")
        print(f"Confidence: {result.get('confidence')}")
    else:
        print("Error occurred")
        
except Exception as e:
    print(f"Error: {e}")
