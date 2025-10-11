from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for model and preprocessors
model = None
scaler = None
label_encoders = {}

def load_model_and_preprocessors():
    """Load the trained model and preprocessors"""
    global model, scaler, label_encoders
    
    try:
        # Try to load saved model and preprocessors
        if os.path.exists('models/rf_model.joblib'):
            model = joblib.load('models/rf_model.joblib')
        else:
            # Create a new Random Forest model if no saved model exists
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
        if os.path.exists('models/scaler.joblib'):
            scaler = joblib.load('models/scaler.joblib')
        else:
            scaler = StandardScaler()
            
        # Load label encoders for categorical features
        categorical_features = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD', 'Blood Pressure Category']
        for feature in categorical_features:
            if os.path.exists(f'models/{feature}_encoder.joblib'):
                label_encoders[feature] = joblib.load(f'models/{feature}_encoder.joblib')
            else:
                label_encoders[feature] = LabelEncoder()
                
    except Exception as e:
        print(f"Error loading model: {e}")
        # Initialize with default values
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()

def preprocess_input(data):
    """Preprocess input data for prediction"""
    try:
        print(f"Input data: {data}")
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        print(f"DataFrame created: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Handle categorical features
        categorical_features = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD', 'Blood Pressure Category']
        
        for feature in categorical_features:
            if feature in df.columns:
                # Use label encoder if available, otherwise use simple mapping
                if feature in label_encoders:
                    try:
                        df[feature] = label_encoders[feature].transform(df[feature].astype(str))
                        print(f"Encoded {feature}: {df[feature].iloc[0]}")
                    except ValueError as e:
                        print(f"Error encoding {feature}: {e}")
                        # Handle unseen categories
                        df[feature] = 0
                else:
                    # Simple mapping for common values
                    if feature == 'Sex':
                        df[feature] = df[feature].map({'M': 1, 'F': 0}).fillna(0)
                    elif feature == 'Smoking Status':
                        df[feature] = df[feature].map({'Y': 1, 'N': 0}).fillna(0)
                    elif feature == 'Diabetes Status':
                        df[feature] = df[feature].map({'Y': 1, 'N': 0}).fillna(0)
                    elif feature == 'Physical Activity Level':
                        df[feature] = df[feature].map({'High': 2, 'Moderate': 1, 'Low': 0}).fillna(0)
                    elif feature == 'Family History of CVD':
                        df[feature] = df[feature].map({'Y': 1, 'N': 0}).fillna(0)
                    elif feature == 'Blood Pressure Category':
                        df[feature] = df[feature].map({'Normal': 0, 'Elevated': 1, 'Hypertension Stage 1': 2, 'Hypertension Stage 2': 3}).fillna(0)
        
        # Calculate derived features
        if 'Height (m)' in df.columns and 'Weight (kg)' in df.columns:
            df['BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)
            print(f"Calculated BMI: {df['BMI'].iloc[0]}")
        
        if 'Height (cm)' in df.columns and 'Abdominal Circumference (cm)' in df.columns:
            df['Waist-to-Height Ratio'] = df['Abdominal Circumference (cm)'] / df['Height (cm)']
        
        if 'Total Cholesterol (mg/dL)' in df.columns and 'HDL (mg/dL)' in df.columns:
            df['Cholesterol_Ratio'] = df['Total Cholesterol (mg/dL)'] / df['HDL (mg/dL)']
        
        if 'Systolic BP' in df.columns and 'Diastolic BP' in df.columns:
            df['BP_Ratio'] = df['Systolic BP'] / df['Diastolic BP']
        
        # Calculate CVD Risk Score (simplified version)
        risk_score = 0
        if 'Age' in df.columns:
            risk_score += df['Age'].iloc[0] * 0.1
        if 'BMI' in df.columns:
            risk_score += df['BMI'].iloc[0] * 0.2
        if 'Systolic BP' in df.columns:
            risk_score += df['Systolic BP'].iloc[0] * 0.05
        if 'Total Cholesterol (mg/dL)' in df.columns:
            risk_score += df['Total Cholesterol (mg/dL)'].iloc[0] * 0.01
        
        df['CVD Risk Score'] = risk_score
        print(f"Calculated CVD Risk Score: {risk_score}")
        
        # Select features for prediction (based on the notebook analysis)
        feature_columns = [
            'Age', 'Weight (kg)', 'Height (m)', 'BMI', 'Abdominal Circumference (cm)',
            'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)',
            'Height (cm)', 'Waist-to-Height Ratio', 'Systolic BP', 'Diastolic BP',
            'Estimated LDL (mg/dL)', 'CVD Risk Score', 'Cholesterol_Ratio', 'BP_Ratio',
            'Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level',
            'Family History of CVD', 'Blood Pressure Category'
        ]
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
                print(f"Added missing feature: {col}")
        
        # Select only the required features
        X = df[feature_columns]
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Feature values: {X.iloc[0].to_dict()}")
        
        # Scale the features
        X_scaled = scaler.transform(X)
        print(f"Scaled features shape: {X_scaled.shape}")
        
        return X_scaled
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_risk_suggestions(risk_level):
    """Get suggestions based on risk level"""
    suggestions = {
        'LOW': {
            'title': 'Low Risk - Keep Up the Good Work!',
            'suggestions': [
                'Continue maintaining a healthy lifestyle',
                'Keep up regular physical activity (at least 150 minutes per week)',
                'Maintain a balanced diet rich in fruits and vegetables',
                'Schedule annual health check-ups',
                'Avoid smoking and limit alcohol consumption',
                'Manage stress through relaxation techniques'
            ],
            'color': 'success'
        },
        'INTERMEDIARY': {
            'title': 'Moderate Risk - Time for Action',
            'suggestions': [
                'Increase physical activity to at least 30 minutes daily',
                'Focus on heart-healthy diet (reduce saturated fats)',
                'Monitor blood pressure and cholesterol regularly',
                'Consider weight management if BMI is elevated',
                'Reduce sodium intake to less than 2,300mg per day',
                'Schedule more frequent health check-ups (every 6 months)',
                'Consider stress management programs'
            ],
            'color': 'warning'
        },
        'HIGH': {
            'title': 'High Risk - Immediate Action Required',
            'suggestions': [
                'Consult with a healthcare provider immediately',
                'Implement aggressive lifestyle modifications',
                'Consider medication under medical supervision',
                'Monitor vital signs daily (blood pressure, weight)',
                'Follow a strict heart-healthy diet plan',
                'Engage in supervised exercise program',
                'Quit smoking immediately if applicable',
                'Schedule frequent medical follow-ups',
                'Consider cardiac rehabilitation program'
            ],
            'color': 'danger'
        }
    }
    return suggestions.get(risk_level, suggestions['LOW'])

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert numeric fields
        numeric_fields = ['Age', 'Weight (kg)', 'Height (m)', 'Height (cm)', 'Abdominal Circumference (cm)',
                         'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)',
                         'Systolic BP', 'Diastolic BP', 'Estimated LDL (mg/dL)']
        
        for field in numeric_fields:
            if field in data and data[field]:
                try:
                    data[field] = float(data[field])
                except ValueError:
                    data[field] = 0.0
            else:
                data[field] = 0.0
        
        # Preprocess the input
        X_scaled = preprocess_input(data)
        
        if X_scaled is None:
            return jsonify({'error': 'Error processing input data'})
        
        # Make prediction
        print(f"Model loaded: {model is not None}")
        print(f"Scaler loaded: {scaler is not None}")
        print(f"Input shape: {X_scaled.shape}")
        
        prediction = model.predict(X_scaled)[0]
        print(f"Prediction: {prediction}")
        
        # Map prediction to risk level
        risk_mapping = {0: 'LOW', 1: 'INTERMEDIARY', 2: 'HIGH'}
        risk_level = risk_mapping.get(prediction, 'LOW')
        
        # Get suggestions
        suggestions = get_risk_suggestions(risk_level)
        
        # Get prediction probability
        try:
            probabilities = model.predict_proba(X_scaled)[0]
            confidence = max(probabilities) * 100
        except:
            confidence = 85.0  # Default confidence
        
        return jsonify({
            'risk_level': risk_level,
            'confidence': round(confidence, 1),
            'suggestions': suggestions
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'})

if __name__ == '__main__':
    # Load model and preprocessors
    load_model_and_preprocessors()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
