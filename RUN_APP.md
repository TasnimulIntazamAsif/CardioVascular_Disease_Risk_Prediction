# Running the CVD Risk Prediction Flask App

## 1. Save the Random Forest Fusion model (one-time)

From the project root, run:

```bash
pip install -r requirements.txt
python save_fusion_model.py
```

This will create the `models/` folder with:
- `rf_weighted_fusion_model.joblib` – Random Forest Fusion model
- `scaler_clinical.joblib`, `scaler_lifestyle.joblib` – feature scalers
- `fusion_params.joblib` – fusion weights and feature lists
- `*_encoder.joblib` – label encoders for categorical inputs

## 2. Start the Flask app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

The app will load the saved Random Forest Fusion model and use it for predictions.
