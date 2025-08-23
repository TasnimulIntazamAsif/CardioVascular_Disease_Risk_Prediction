# XGBoost Model for Cardiovascular Disease Risk Prediction
# Complete implementation with data preparation, model fitting, and prediction

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")

# Load the dataset
print("Loading dataset...")
try:
    # Try to load the dataset
    data = pd.read_csv('CVD_Dataset.csv')
    print(f"Dataset loaded successfully! Shape: {data.shape}")
except FileNotFoundError:
    print("CVD_Dataset.csv not found. Please ensure the dataset file is in the current directory.")
    print("You can download the dataset or provide the correct path.")
    exit()

# Display basic information about the dataset
print("\nDataset Info:")
print(data.info())
print("\nFirst few rows:")
print(data.head())
print("\nDataset shape:", data.shape)

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Check target variable distribution
if 'target' in data.columns:
    target_col = 'target'
elif 'cardio' in data.columns:
    target_col = 'cardio'
else:
    # Try to identify the target column
    target_col = None
    for col in data.columns:
        if data[col].dtype in ['int64', 'float64'] and data[col].nunique() <= 2:
            target_col = col
            break

if target_col:
    print(f"\nTarget variable: {target_col}")
    print("Target distribution:")
    print(data[target_col].value_counts())
    print(f"Target distribution percentage:")
    print(data[target_col].value_counts(normalize=True) * 100)
else:
    print("Could not identify target variable. Please specify the target column.")
    exit()

# Data Preprocessing
print("\n=== Data Preprocessing ===")

# Separate features and target
X = data.drop(columns=[target_col])
y = data[target_col]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Handle categorical variables
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

print(f"\nCategorical columns: {list(categorical_cols)}")
print(f"Numerical columns: {list(numerical_cols)}")

# Encode categorical variables
if len(categorical_cols) > 0:
    print("\nEncoding categorical variables...")
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])
    print("Categorical variables encoded successfully!")

# Scale numerical features
if len(numerical_cols) > 0:
    print("\nScaling numerical features...")
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    print("Numerical features scaled successfully!")

# Split the data
print("\n=== Data Splitting ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Training target distribution: {np.bincount(y_train)}")
print(f"Testing target distribution: {np.bincount(y_test)}")

# XGBoost Model Fitting
print("\n=== XGBoost Model Fitting ===")

# Initialize XGBoost classifier
xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    use_label_encoder=False
)

print("Initial XGBoost model created!")

# Hyperparameter tuning with Grid Search
print("\nPerforming hyperparameter tuning...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}

# Grid Search with Cross-Validation
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit the grid search
print("Fitting Grid Search (this may take a while)...")
grid_search.fit(X_train, y_train)

# Get best parameters
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Get the best model
best_xgb_model = grid_search.best_estimator_
print("Best XGBoost model obtained!")

# Model Training
print("\n=== Final Model Training ===")
print("Training the best model on full training data...")

# Train the best model on full training data
best_xgb_model.fit(X_train, y_train)

print("Model training completed!")

# Model Evaluation
print("\n=== Model Evaluation ===")

# Make predictions
y_pred = best_xgb_model.predict(X_test)
y_pred_proba = best_xgb_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature Importance
print("\n=== Feature Importance ===")
feature_importance = best_xgb_model.feature_importances_
feature_names = X.columns

# Create feature importance DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(importance_df.head(10))

# Cross-validation scores
print("\n=== Cross-Validation Scores ===")
cv_scores = cross_val_score(best_xgb_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Model Performance Visualization
print("\n=== Creating Performance Visualizations ===")

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('XGBoost Model Performance Analysis', fontsize=16)

# 1. Feature Importance Plot
axes[0, 0].barh(range(len(importance_df.head(15))), importance_df.head(15)['importance'])
axes[0, 0].set_yticks(range(len(importance_df.head(15))))
axes[0, 0].set_yticklabels(importance_df.head(15)['feature'])
axes[0, 0].set_xlabel('Feature Importance')
axes[0, 0].set_title('Top 15 Feature Importances')
axes[0, 0].invert_yaxis()

# 2. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1, 0].set_xlim([0.0, 1.0])
axes[1, 0].set_ylim([0.0, 1.05])
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].legend(loc="lower right")

# 4. Cross-validation scores
axes[1, 1].bar(range(1, 6), cv_scores, color='skyblue', alpha=0.7)
axes[1, 1].axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.3f}')
axes[1, 1].set_xlabel('Fold')
axes[1, 1].set_ylabel('Accuracy Score')
axes[1, 1].set_title('Cross-Validation Scores')
axes[1, 1].legend()
axes[1, 1].set_xticks(range(1, 6))

plt.tight_layout()
plt.show()

# Save the model
print("\n=== Saving the Model ===")
import joblib

# Save the best model
joblib.dump(best_xgb_model, 'xgb_best_model.joblib')
print("Best XGBoost model saved as 'xgb_best_model.joblib'")

# Save the scaler and label encoders
joblib.dump(scaler, 'xgb_scaler.joblib')
if len(categorical_cols) > 0:
    joblib.dump(label_encoders, 'xgb_label_encoders.joblib')
    print("Scaler and label encoders saved!")

# Save feature names
joblib.dump(feature_names.tolist(), 'xgb_feature_names.joblib')
print("Feature names saved!")

# Model Summary
print("\n=== Model Summary ===")
print(f"Dataset: {data.shape[0]} samples, {data.shape[1]-1} features")
print(f"Target variable: {target_col}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Best accuracy: {accuracy:.4f}")
print(f"Best ROC AUC: {roc_auc:.4f}")
print(f"Best parameters: {grid_search.best_params_}")

print("\nXGBoost model training and evaluation completed successfully!")
print("The model is ready for predictions on new data.")
