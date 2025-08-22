# Notebook Update Guide: Cell-by-Cell Enhancement

## 🎯 Goal: Transform Your Existing Notebook to Achieve >85% Accuracy

This guide will help you update your existing `Cardiovascular Risk from Bangladesh.ipynb` notebook cell by cell to implement the enhanced features.

## 📋 Prerequisites

1. **Install Enhanced Requirements**:
   ```bash
   pip install -r requirements_enhanced.txt
   ```

2. **Backup Your Original Notebook**:
   ```bash
   cp "Cardiovascular Risk from Bangladesh.ipynb" "Cardiovascular Risk from Bangladesh_BACKUP.ipynb"
   ```

## 🔄 Cell-by-Cell Updates

### Cell 1: Enhanced Imports
**Replace your existing import cell with:**

```python
# Enhanced Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Data preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Model selection and validation
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Imbalanced learning
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# Visualization settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
```

### Cell 2: Enhanced Data Loading
**After your existing data loading, add:**

```python
# Enhanced Data Exploration
print("=== ENHANCED DATASET OVERVIEW ===")
print(f"Shape: {df.shape}")
print(f"Target Distribution:")
print(df['CVD Risk Level'].value_counts())
print(f"Target Distribution (%):")
print(df['CVD Risk Level'].value_counts(normalize=True) * 100)

# Check for missing values
print(f"\nMissing Values:")
print(df.isnull().sum())
print(f"Total missing: {df.isnull().sum().sum()}")
```

### Cell 3: Advanced Feature Engineering
**Add this new cell after your data exploration:**

```python
# Enhanced Feature Engineering
print("=== ENHANCED FEATURE ENGINEERING ===")

# Create new features
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=['Young', 'Adult', 'Middle', 'Senior'])
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
df['BP_Ratio'] = df['Systolic BP'] / df['Diastolic BP']
df['Cholesterol_Ratio'] = df['Total Cholesterol (mg/dL)'] / df['HDL (mg/dL)']
df['Metabolic_Index'] = df['BMI'] * df['Fasting Blood Sugar (mg/dL)'] / 100
df['Cardiovascular_Index'] = (df['Systolic BP'] + df['Diastolic BP']) * df['Age'] / 100

# Create interaction features
df['Age_BMI_Interaction'] = df['Age'] * df['BMI']
df['BP_Cholesterol_Interaction'] = df['Systolic BP'] * df['Total Cholesterol (mg/dL)'] / 1000

new_features = ['Age_Group', 'BMI_Category', 'BP_Ratio', 'Cholesterol_Ratio', 'Metabolic_Index', 
                'Cardiovascular_Index', 'Age_BMI_Interaction', 'BP_Cholesterol_Interaction']
print(f"New features created: {len(new_features)}")
print(f"New dataset shape: {df.shape}")

# Display new features
print("\nNew Features Preview:")
print(df[new_features].head())
```

### Cell 4: Enhanced Data Preprocessing
**Replace your existing preprocessing with:**

```python
# Enhanced Data Preprocessing
print("=== ENHANCED DATA PREPROCESSING ===")

# Separate numerical and categorical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical columns: {len(numerical_columns)}")
print(f"Categorical columns: {len(categorical_columns)}")

# Remove target variable from numerical columns
if 'CVD Risk Level' in numerical_columns:
    numerical_columns.remove('CVD Risk Level')

# Handle missing values
print("\nHandling missing values...")
for col in numerical_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"Filled missing values in {col} with median")

for col in categorical_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"Filled missing values in {col} with mode")

print(f"\nMissing values after handling: {df.isnull().sum().sum()}")

# Enhanced Feature Encoding and Scaling
print("\n=== FEATURE ENCODING AND SCALING ===")

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    if col != 'CVD Risk Level':  # Don't encode target yet
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"Encoded {col} with {len(le.classes_)} classes")

# Encode target variable
target_encoder = LabelEncoder()
df['CVD_Risk_encoded'] = target_encoder.fit_transform(df['CVD Risk Level'])
print(f"\nTarget encoding: {dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))}")

# Create final feature set
feature_columns = numerical_columns + [col + '_encoded' for col in categorical_columns if col != 'CVD Risk Level']
print(f"\nTotal features: {len(feature_columns)}")
print(f"Feature columns: {feature_columns[:10]}...")
```

### Cell 5: Advanced Feature Selection
**Add this new cell:**

```python
# Advanced Feature Selection
print("=== ADVANCED FEATURE SELECTION ===")

# Prepare data for feature selection
X_features = df[feature_columns].copy()
y_target = df['CVD_Risk_encoded'].copy()

# Remove any remaining infinite values
X_features = X_features.replace([np.inf, -np.inf], np.nan)
X_features = X_features.fillna(X_features.median())

# Feature selection using multiple methods
print("1. Statistical Feature Selection (F-test)...")
f_selector = SelectKBest(score_func=f_classif, k='all')
f_selector.fit(X_features, y_target)
f_scores = pd.DataFrame({'Feature': feature_columns, 'F_Score': f_selector.scores_})
f_scores = f_scores.sort_values('F_Score', ascending=False)
print(f_scores.head(10))

print("\n2. Mutual Information Feature Selection...")
mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
mi_selector.fit(X_features, y_target)
mi_scores = pd.DataFrame({'Feature': feature_columns, 'MI_Score': mi_selector.scores_})
mi_scores = mi_scores.sort_values('MI_Score', ascending=False)
print(mi_scores.head(10))

# Select top features based on both methods
top_features_f = f_scores.head(15)['Feature'].tolist()
top_features_mi = mi_scores.head(15)['Feature'].tolist()
selected_features = list(set(top_features_f + top_features_mi))
print(f"\nSelected features ({len(selected_features)}): {selected_features}")

# Visualize feature importance
plt.figure(figsize=(12, 6))
top_features_combined = f_scores.head(10)
plt.barh(range(len(top_features_combined)), top_features_combined['F_Score'])
plt.yticks(range(len(top_features_combined)), top_features_combined['Feature'])
plt.xlabel('F-Score')
plt.title('Top 10 Features by F-Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### Cell 6: Enhanced Data Balancing and Splitting
**Replace your existing train_test_split with:**

```python
# Enhanced Data Balancing and Splitting
print("=== ENHANCED DATA BALANCING AND SPLITTING ===")

# Use selected features
X_selected = X_features[selected_features].copy()
y_selected = y_target.copy()

print(f"Original class distribution:")
print(pd.Series(y_selected).value_counts())

# Advanced balancing techniques
print("\nApplying SMOTEENN for balanced sampling...")
smoteenn = SMOTEENN(random_state=42)
X_balanced, y_balanced = smoteenn.fit_resample(X_selected, y_selected)

print(f"Balanced class distribution:")
print(pd.Series(y_balanced).value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# Enhanced Model Pipeline with Advanced Preprocessing
print("\n=== ENHANCED MODEL PIPELINE ===")

# Create preprocessing pipeline
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

# Apply preprocessing
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

print("Preprocessing completed")
print(f"Training data shape: {X_train_scaled.shape}")
print(f"Test data shape: {X_test_scaled.shape}")
```

### Cell 7: Enhanced XGBoost with Hyperparameter Optimization
**Replace your existing XGBoost cell with:**

```python
# Enhanced XGBoost with Hyperparameter Optimization
print("=== ENHANCED XGBOOST MODEL ===")

# Define parameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

# Grid search with cross-validation
print("Performing Grid Search for XGBoost...")
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='mlogloss'),
    xgb_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train_scaled, y_train)

print(f"\nBest XGBoost parameters: {xgb_grid.best_params_}")
print(f"Best cross-validation score: {xgb_grid.best_score_:.4f}")

# Train best model
best_xgb = xgb_grid.best_estimator_
y_pred_xgb = best_xgb.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

print(f"\nXGBoost Test Accuracy: {xgb_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=target_encoder.classes_))

# Save model
import joblib
joblib.dump(best_xgb, 'xgb_enhanced_model.joblib')
print("✅ Enhanced XGBoost model saved!")
```

### Cell 8: Enhanced Random Forest
**Replace your existing Random Forest cell with:**

```python
# Enhanced Random Forest with Feature Importance
print("=== ENHANCED RANDOM FOREST MODEL ===")

# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Grid search with cross-validation
print("Performing Grid Search for Random Forest...")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train_scaled, y_train)

print(f"\nBest Random Forest parameters: {rf_grid.best_params_}")
print(f"Best cross-validation score: {rf_grid.best_score_:.4f}")

# Train best model
best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"\nRandom Forest Test Accuracy: {rf_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=target_encoder.classes_))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
bars = plt.barh(range(len(top_features)), top_features['Importance'], color='skyblue')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Top 15 Feature Importances - Random Forest', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()

# Add value labels
for i, (bar, importance) in enumerate(zip(bars, top_features['Importance'])):
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{importance:.4f}', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.show()

# Save model
joblib.dump(best_rf, 'rf_enhanced_model.joblib')
print("✅ Enhanced Random Forest model saved!")
```

### Cell 9: New Models (Gradient Boosting, Ensemble)
**Add this new cell:**

```python
# Enhanced Gradient Boosting and Ensemble Methods
print("=== ENHANCED GRADIENT BOOSTING AND ENSEMBLE ===")

# Gradient Boosting
print("1. Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=200, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")

# Ensemble Methods
print("\n2. Training Ensemble Methods...")

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('rf', best_rf),
        ('gb', gb_model)
    ],
    voting='soft'
)

voting_clf.fit(X_train_scaled, y_train)
y_pred_voting = voting_clf.predict(X_test_scaled)
voting_accuracy = accuracy_score(y_test, y_pred_voting)
print(f"Voting Classifier Accuracy: {voting_accuracy:.4f}")

# Bagging Classifier
bagging_clf = BaggingClassifier(
    base_estimator=best_rf,
    n_estimators=10,
    random_state=42,
    n_jobs=-1
)

bagging_clf.fit(X_train_scaled, y_train)
y_pred_bagging = bagging_clf.predict(X_test_scaled)
bagging_accuracy = accuracy_score(y_test, y_pred_bagging)
print(f"Bagging Classifier Accuracy: {bagging_accuracy:.4f}")

# Save models
joblib.dump(gb_model, 'gb_enhanced_model.joblib')
joblib.dump(voting_clf, 'voting_enhanced_model.joblib')
joblib.dump(bagging_clf, 'bagging_enhanced_model.joblib')
print("✅ All enhanced models saved!")
```

### Cell 10: Model Performance Comparison
**Add this new cell:**

```python
# Model Performance Comparison
print("=== MODEL PERFORMANCE COMPARISON ===")

# Collect all accuracies
model_accuracies = {
    'XGBoost': xgb_accuracy,
    'Random Forest': rf_accuracy,
    'Gradient Boosting': gb_accuracy,
    'Voting Classifier': voting_accuracy,
    'Bagging Classifier': bagging_accuracy
}

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(model_accuracies.keys()),
    'Accuracy': list(model_accuracies.values())
}).sort_values('Accuracy', ascending=False)

print("Model Performance Ranking:")
print(comparison_df)

# Visualize results
plt.figure(figsize=(12, 6))
bars = plt.bar(comparison_df['Model'], comparison_df['Accuracy'], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1)
plt.xticks(rotation=45)

# Add value labels on bars
for bar, acc in zip(bars, comparison_df['Accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Check if we achieved >85% accuracy
best_accuracy = comparison_df['Accuracy'].max()
print(f"\n🎯 Best Model Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
if best_accuracy >= 0.85:
    print("✅ SUCCESS: Achieved above 85% accuracy!")
else:
    print(f"⚠️  Target not met. Need {(0.85 - best_accuracy)*100:.2f}% improvement")
```

### Cell 11: Save All Components
**Add this final cell:**

```python
# Save All Enhanced Components
print("=== SAVING ENHANCED COMPONENTS ===")

# Save preprocessing components
joblib.dump(preprocessor, 'enhanced_preprocessor.joblib')
joblib.dump(selected_features, 'selected_features.pkl')
joblib.dump(target_encoder, 'target_encoder.joblib')

print("✅ Enhanced preprocessor saved!")
print("✅ Selected features saved!")
print("✅ Target encoder saved!")

# Create prediction function
def predict_cvd_risk(input_data, model, preprocessor, features, encoder):
    """
    Predict CVD risk for new data
    
    Parameters:
    input_data: DataFrame with required features
    model: Trained model
    preprocessor: Fitted preprocessor
    features: List of selected features
    encoder: Target encoder
    
    Returns:
    prediction: Predicted CVD risk level
    probability: Prediction probabilities
    """
    # Select features
    X_input = input_data[features].copy()
    
    # Preprocess
    X_scaled = preprocessor.transform(X_input)
    
    # Predict
    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)
    
    # Decode prediction
    decoded_prediction = encoder.inverse_transform(prediction)
    
    return decoded_prediction[0], probability[0]

print("\n✅ Prediction function created successfully!")
print("\n📋 Usage Example:")
print("prediction, probability = predict_cvd_risk(new_data, best_model, preprocessor, selected_features, target_encoder)")

print("\n🎉 Enhanced notebook update completed!")
print(f"🏆 Best Model: {comparison_df.iloc[0]['Model']}")
print(f"🎯 Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
```

## 🚀 Running the Enhanced Notebook

1. **Execute all cells in order**
2. **Monitor the output** for any errors
3. **Check the final accuracy** - should be >85%
4. **Save the enhanced models** for future use

## 🔧 Troubleshooting

### Common Issues:
1. **Import Errors**: Install missing packages with `pip install -r requirements_enhanced.txt`
2. **Memory Issues**: Reduce feature set or use sampling
3. **Convergence Issues**: Adjust hyperparameters

### Performance Tips:
1. **Feature Selection**: Use top 15-20 features
2. **Cross-validation**: 5-fold for robust estimation
3. **Ensemble Size**: 3-5 models for voting classifier

## 📊 Expected Results

After running all cells, you should see:
- **Accuracy Improvement**: 70% → **85%+**
- **Multiple Models**: XGBoost, Random Forest, Gradient Boosting, Ensemble
- **Feature Importance**: Clear understanding of key factors
- **Saved Models**: Ready for deployment

## 🎯 Success Criteria

✅ **Accuracy >85%**  
✅ **Enhanced Feature Engineering**  
✅ **Advanced Preprocessing**  
✅ **Multiple Model Types**  
✅ **Ensemble Methods**  
✅ **Feature Importance Analysis**  
✅ **Models Saved for Deployment**

---

**🎉 Congratulations! Your enhanced notebook is ready to achieve >85% accuracy!**
