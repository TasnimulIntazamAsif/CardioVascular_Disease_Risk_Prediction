import joblib
import numpy as np

# Test loading the weighted fusion model
try:
    print("Testing model loading...")
    
    # Load model
    model = joblib.load('models/rf_weighted_fusion_model.joblib')
    print(f"Model loaded: {type(model)}")
    
    # Load scalers
    scaler_clinical = joblib.load('models/scaler_clinical.joblib')
    scaler_lifestyle = joblib.load('models/scaler_lifestyle.joblib')
    print(f"Scalers loaded")
    
    # Load fusion params
    fusion_params = joblib.load('models/fusion_params.joblib')
    print(f"Fusion params: {fusion_params}")
    
    # Test prediction with dummy data
    dummy_data = np.random.randn(1, 17)  # 13 clinical + 4 lifestyle
    prediction = model.predict(dummy_data)
    print(f"Test prediction: {prediction}")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
