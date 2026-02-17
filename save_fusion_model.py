"""
Train and save the best model (Random Forest Fusion) for the Flask CVD Risk app.

Aligned with app.py:
- Model is loaded from models/rf_weighted_fusion_model.joblib (BEST_MODEL_PATH in app).
- Feature order in fusion_params must match app preprocess_input() (clinical then lifestyle).
- Same clinical/lifestyle lists and fusion weights used for training and inference.

Run once: python save_fusion_model.py
Then run the app: python app.py
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
try:
    from imblearn.combine import SMOTEENN
    HAS_SMOTEENN = True
except ImportError:
    HAS_SMOTEENN = False

# Must match app.py default fusion_params and preprocess_input() feature order
CLINICAL_FEATURES = [
    'Age', 'BMI', 'Weight (kg)', 'Height (m)', 'Systolic BP', 'Diastolic BP',
    'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Estimated LDL (mg/dL)',
    'Fasting Blood Sugar (mg/dL)', 'Abdominal Circumference (cm)',
    'Height (cm)', 'Waist-to-Height Ratio'
]
LIFESTYLE_FEATURES = [
    'Physical Activity Level', 'Smoking Status', 'Diabetes Status', 'Family History of CVD'
]
WEIGHT_CLINICAL = 1.3043848942635585
WEIGHT_LIFESTYLE = 1.182791185390602

# Must match app.py categorical list for label encoders
CATEGORICAL_FOR_ENCODING = [
    'Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level',
    'Family History of CVD', 'Blood Pressure Category'
]

OUT_DIR = 'models'
BEST_MODEL_FILENAME = 'rf_weighted_fusion_model.joblib'


def load_and_prepare_data(csv_path='CVD Dataset.csv'):
    """Load CSV, encode categoricals, compute derived features."""
    df = pd.read_csv(csv_path)
    df.dropna(subset=['Weight (kg)', 'Height (m)', 'Age'], inplace=True)

    # Encode categoricals (same as app will use at prediction time)
    label_encoders = {}
    for col in CATEGORICAL_FOR_ENCODING:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = df[col].astype(str).fillna('Unknown')
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Derived features
    if 'Height (m)' in df.columns and 'Weight (kg)' in df.columns:
        h2 = df['Height (m)'] ** 2
        h2 = h2.replace(0, 1)
        df['BMI'] = df['Weight (kg)'] / h2
    if 'Height (cm)' in df.columns and 'Abdominal Circumference (cm)' in df.columns:
        h = df['Height (cm)'].replace(0, 1)
        df['Waist-to-Height Ratio'] = df['Abdominal Circumference (cm)'] / h
    df['BMI'] = df['BMI'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Waist-to-Height Ratio'] = df['Waist-to-Height Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Target: CVD Risk Level -> 0 LOW, 1 INTERMEDIARY, 2 HIGH
    if 'CVD Risk Level' in df.columns:
        risk_map = {'LOW': 0, 'INTERMEDIARY': 1, 'HIGH': 2}
        df['CVD_Risk_encoded'] = df['CVD Risk Level'].map(risk_map)
        df = df.dropna(subset=['CVD_Risk_encoded'])
    else:
        raise ValueError("Dataset must contain 'CVD Risk Level' column")

    return df, label_encoders


def build_fused_features(Xc, Xl, w_c, w_l):
    """Weight and concatenate clinical and lifestyle features (same formula as app.py)."""
    Xc_w = Xc * w_c
    Xl_w = Xl * w_l
    return np.hstack([Xc_w, Xl_w])


def save_best_model(rf_model, scaler_clinical, scaler_lifestyle, fusion_params, label_encoders, out_dir=OUT_DIR):
    """
    Save the best model (Random Forest Fusion) and all artifacts for the Flask app.
    Call this after training, or use main() to train and save in one go.
    fusion_params must have: weight_clinical, weight_lifestyle, clinical_features, lifestyle_features.
    Feature lists order must match app.py preprocess_input().
    """
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(rf_model, os.path.join(out_dir, BEST_MODEL_FILENAME))
    joblib.dump(scaler_clinical, os.path.join(out_dir, 'scaler_clinical.joblib'))
    joblib.dump(scaler_lifestyle, os.path.join(out_dir, 'scaler_lifestyle.joblib'))
    joblib.dump(fusion_params, os.path.join(out_dir, 'fusion_params.joblib'))
    for name, le in label_encoders.items():
        joblib.dump(le, os.path.join(out_dir, f'{name}_encoder.joblib'))
    print(f"Saved best model (Random Forest Fusion) to {out_dir}/")
    print(f"  - {BEST_MODEL_FILENAME}")
    print(f"  - scaler_clinical.joblib, scaler_lifestyle.joblib")
    print(f"  - fusion_params.joblib")
    print(f"  - *_encoder.joblib ({len(label_encoders)} encoders)")


def main():
    print("Loading data...")
    csv_path = 'CVD Dataset.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}. Place 'CVD Dataset.csv' in the project root.")
    df, label_encoders = load_and_prepare_data(csv_path)

    # Build clinical and lifestyle matrices (only columns that exist)
    clinical_cols = [c for c in CLINICAL_FEATURES if c in df.columns]
    lifestyle_cols = [c for c in LIFESTYLE_FEATURES if c in df.columns]
    if len(clinical_cols) != len(CLINICAL_FEATURES) or len(lifestyle_cols) != len(LIFESTYLE_FEATURES):
        print("Warning: Some features missing. Using:", clinical_cols, lifestyle_cols)
    X_clinical = df[clinical_cols].fillna(0).values
    X_lifestyle = df[lifestyle_cols].fillna(0).values
    y = df['CVD_Risk_encoded'].values

    # Scale
    scaler_clinical = StandardScaler()
    scaler_lifestyle = StandardScaler()
    Xc = scaler_clinical.fit_transform(X_clinical)
    Xl = scaler_lifestyle.fit_transform(X_lifestyle)

    # Fuse
    X_fused = build_fused_features(Xc, Xl, WEIGHT_CLINICAL, WEIGHT_LIFESTYLE)

    # Balance and split
    if HAS_SMOTEENN:
        try:
            smoteenn = SMOTEENN(random_state=42, n_jobs=-1)
            X_bal, y_bal = smoteenn.fit_resample(X_fused, y)
        except Exception:
            X_bal, y_bal = X_fused, y
    else:
        X_bal, y_bal = X_fused, y
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    # Train Random Forest Fusion model (best model for the app)
    print("Training Random Forest Fusion model...")
    rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
    rf_model.fit(X_train, y_train)
    acc = (rf_model.predict(X_test) == y_test).mean()
    print(f"Test accuracy: {acc*100:.2f}%")

    # Fusion params: order must match app.py preprocess_input()
    fusion_params = {
        'weight_clinical': WEIGHT_CLINICAL,
        'weight_lifestyle': WEIGHT_LIFESTYLE,
        'clinical_features': clinical_cols,
        'lifestyle_features': lifestyle_cols,
    }
    save_best_model(rf_model, scaler_clinical, scaler_lifestyle, fusion_params, label_encoders, out_dir=OUT_DIR)
    print("You can now run: python app.py")
    return 0


if __name__ == '__main__':
    main()
