# CVD Risk Assessment Web Application

A Flask-based web application for Cardiovascular Disease (CVD) Risk Assessment using machine learning. This application allows users to input their health information and receive personalized CVD risk level predictions along with recommendations.

## Features

- **Real-time Risk Assessment**: Input health data and get instant CVD risk level prediction
- **Three Risk Levels**: LOW, INTERMEDIARY, and HIGH risk classifications
- **Personalized Recommendations**: Tailored suggestions based on risk level
- **Healthcare-themed UI**: Professional medical interface with highlighting features
- **Responsive Design**: Works on desktop and mobile devices
- **Form Validation**: Real-time validation of input fields
- **Interactive Features**: BMI auto-calculation, height conversion, etc.

## Risk Levels

- **LOW**: Low cardiovascular disease risk
- **INTERMEDIARY**: Moderate cardiovascular disease risk  
- **HIGH**: High cardiovascular disease risk

## Required Input Features

### Personal Information
- Age (years)
- Sex (Male/Female)

### Physical Measurements
- Weight (kg)
- Height (meters/cm)
- Abdominal Circumference (cm)
- BMI (auto-calculated)

### Blood Pressure
- Systolic BP (mmHg)
- Diastolic BP (mmHg)
- Blood Pressure Category

### Cholesterol Levels
- Total Cholesterol (mg/dL)
- HDL Cholesterol (mg/dL)
- Estimated LDL (mg/dL)

### Blood Sugar
- Fasting Blood Sugar (mg/dL)

### Lifestyle Factors
- Smoking Status
- Diabetes Status
- Physical Activity Level
- Family History of CVD

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Project Structure

```
CardioVascular_Disease_Risk_Prediction/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── style.css         # CSS styling
│   └── script.js         # JavaScript functionality
├── models/               # Directory for saved models (created automatically)
└── CVD Dataset.csv       # Training dataset
```

## Usage

1. **Fill out the assessment form** with your health information
2. **Required fields** are marked with asterisks (*)
3. **Click "Calculate Risk Assessment"** to get your results
4. **View your risk level** and personalized recommendations
5. **Use "New Assessment"** to start over

## Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: 22 health-related features including clinical and lifestyle factors
- **Training**: Based on CVD dataset with 1,518 samples
- **Accuracy**: High accuracy model trained on comprehensive health data

## Recommendations by Risk Level

### LOW Risk
- Continue maintaining healthy lifestyle
- Regular physical activity
- Balanced diet
- Annual health check-ups

### INTERMEDIARY Risk
- Increase physical activity
- Heart-healthy diet modifications
- Regular monitoring
- Weight management if needed

### HIGH Risk
- Immediate medical consultation
- Aggressive lifestyle modifications
- Medication consideration
- Frequent medical follow-ups

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **ML Library**: scikit-learn
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib

## Important Notes

⚠️ **Medical Disclaimer**: This tool is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## Browser Compatibility

- Chrome (recommended)
- Firefox
- Safari
- Edge

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port in `app.py` (line 200)
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Model loading errors**: The app will create a default model if saved models are not found

### Support

For technical issues or questions about the application, please check the console output for error messages.

## Future Enhancements

- User authentication and data persistence
- Advanced visualization of risk factors
- Integration with wearable devices
- Multi-language support
- API endpoints for mobile apps

---

**Developed with ❤️ for cardiovascular health awareness**
