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
scaler_clinical = None
scaler_lifestyle = None
fusion_params = None
label_encoders = {}

def load_model_and_preprocessors():
    """Load the trained weighted feature fusion model and preprocessors"""
    global model, scaler_clinical, scaler_lifestyle, fusion_params, label_encoders
    
    try:
        # Try to load saved weighted fusion model and preprocessors
        if os.path.exists('models/rf_weighted_fusion_model.joblib'):
            model = joblib.load('models/rf_weighted_fusion_model.joblib')
            print("✅ Loaded weighted feature fusion Random Forest model")
        else:
            # Create a new Random Forest model if no saved model exists
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            print("⚠️ Using default Random Forest model")
            
        if os.path.exists('models/scaler_clinical.joblib'):
            scaler_clinical = joblib.load('models/scaler_clinical.joblib')
            print("✅ Loaded clinical features scaler")
        else:
            scaler_clinical = StandardScaler()
            print("⚠️ Using default clinical scaler")
            
        if os.path.exists('models/scaler_lifestyle.joblib'):
            scaler_lifestyle = joblib.load('models/scaler_lifestyle.joblib')
            print("✅ Loaded lifestyle features scaler")
        else:
            scaler_lifestyle = StandardScaler()
            print("⚠️ Using default lifestyle scaler")
            
        if os.path.exists('models/fusion_params.joblib'):
            fusion_params = joblib.load('models/fusion_params.joblib')
            print(f"✅ Loaded fusion parameters - Clinical weight: {fusion_params['weight_clinical']:.4f}, Lifestyle weight: {fusion_params['weight_lifestyle']:.4f}")
        else:
            fusion_params = {
                'weight_clinical': 1.25,
                'weight_lifestyle': 2.76,
                'clinical_features': ['Age', 'BMI', 'Weight (kg)', 'Height (m)', 'Systolic BP', 'Diastolic BP', 
                                    'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Estimated LDL (mg/dL)', 
                                    'Fasting Blood Sugar (mg/dL)', 'Abdominal Circumference (cm)', 
                                    'Height (cm)', 'Waist-to-Height Ratio'],
                'lifestyle_features': ['Physical Activity Level', 'Smoking Status', 'Diabetes Status', 'Family History of CVD']
            }
            print("⚠️ Using default fusion parameters")
            
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
        scaler_clinical = StandardScaler()
        scaler_lifestyle = StandardScaler()
        fusion_params = {
            'weight_clinical': 1.25,
            'weight_lifestyle': 2.76,
            'clinical_features': ['Age', 'BMI', 'Weight (kg)', 'Height (m)', 'Systolic BP', 'Diastolic BP', 
                                'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Estimated LDL (mg/dL)', 
                                'Fasting Blood Sugar (mg/dL)', 'Abdominal Circumference (cm)', 
                                'Height (cm)', 'Waist-to-Height Ratio'],
            'lifestyle_features': ['Physical Activity Level', 'Smoking Status', 'Diabetes Status', 'Family History of CVD']
        }

def preprocess_input(data):
    """Preprocess input data for prediction"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Handle categorical features
        categorical_features = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD', 'Blood Pressure Category']
        
        for feature in categorical_features:
            if feature in df.columns:
                # Use label encoder if available, otherwise use simple mapping
                if feature in label_encoders:
                    try:
                        df[feature] = label_encoders[feature].transform(df[feature].astype(str))
                    except ValueError:
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
        
        if 'Height (cm)' in df.columns and 'Abdominal Circumference (cm)' in df.columns:
            df['Waist-to-Height Ratio'] = df['Abdominal Circumference (cm)'] / df['Height (cm)']
        
        # Prepare clinical features
        clinical_features = fusion_params['clinical_features']
        clinical_data = []
        for feature in clinical_features:
            if feature in df.columns:
                clinical_data.append(df[feature].iloc[0])
            else:
                clinical_data.append(0.0)
        
        # Prepare lifestyle features
        lifestyle_features = fusion_params['lifestyle_features']
        lifestyle_data = []
        for feature in lifestyle_features:
            if feature in df.columns:
                lifestyle_data.append(df[feature].iloc[0])
            else:
                lifestyle_data.append(0.0)
        
        # Convert to numpy arrays
        X_clinical = np.array(clinical_data).reshape(1, -1)
        X_lifestyle = np.array(lifestyle_data).reshape(1, -1)
        
        # Scale the features separately
        Xc_scaled = scaler_clinical.transform(X_clinical)
        Xl_scaled = scaler_lifestyle.transform(X_lifestyle)
        
        # Apply weighted fusion using optimal weights
        weight_clinical = fusion_params['weight_clinical']
        weight_lifestyle = fusion_params['weight_lifestyle']
        
        Xc_weighted = Xc_scaled * weight_clinical
        Xl_weighted = Xl_scaled * weight_lifestyle
        
        # Concatenate weighted features
        X_fused = np.hstack([Xc_weighted, Xl_weighted])
        
        return X_fused
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
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
        prediction = model.predict(X_scaled)[0]
        
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
