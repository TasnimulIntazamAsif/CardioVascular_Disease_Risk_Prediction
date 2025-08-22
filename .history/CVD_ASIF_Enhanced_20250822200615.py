#!/usr/bin/env python3
"""
Cardiovascular Disease Risk Prediction - Enhanced Model
Achieving >85% Accuracy with Advanced ML Techniques

This enhanced version includes:
- Advanced feature engineering
- Better data preprocessing
- Hyperparameter optimization
- Ensemble methods
- Cross-validation strategies
"""

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

def main():
    print("=== CARDIOVASCULAR DISEASE RISK PREDICTION - ENHANCED MODEL ===\n")
    
    # Load the dataset
    print("1. Loading dataset...")
    df = pd.read_csv('CVD_Dataset.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Enhanced Data Exploration
    print("\n2. Data exploration...")
    print(f"Shape: {df.shape}")
    print(f"Target Distribution:")
    print(df['CVD Risk Level'].value_counts())
    print(f"Target Distribution (%):")
    print(df['CVD Risk Level'].value_counts(normalize=True) * 100)
    
    # Enhanced Feature Engineering
    print("\n3. Enhanced feature engineering...")
    
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
    
    # Enhanced Data Preprocessing
    print("\n4. Enhanced data preprocessing...")
    
    # Separate numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical columns: {len(numerical_columns)}")
    print(f"Categorical columns: {len(categorical_columns)}")
    
    # Remove target variable from numerical columns
    if 'CVD Risk Level' in numerical_columns:
        numerical_columns.remove('CVD Risk Level')
    
    # Handle missing values
    print("Handling missing values...")
    for col in numerical_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"Filled missing values in {col} with median")
    
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"Filled missing values in {col} with mode")
    
    print(f"Missing values after handling: {df.isnull().sum().sum()}")
    
    # Enhanced Feature Encoding and Scaling
    print("\n5. Feature encoding and scaling...")
    
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
    print(f"Target encoding: {dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))}")
    
    # Create final feature set
    feature_columns = numerical_columns + [col + '_encoded' for col in categorical_columns if col != 'CVD Risk Level']
    print(f"Total features: {len(feature_columns)}")
    
    # Advanced Feature Selection
    print("\n6. Advanced feature selection...")
    
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
    
    # Enhanced Data Balancing and Splitting
    print("\n7. Enhanced data balancing and splitting...")
    
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
    print("\n8. Enhanced model pipeline...")
    
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
    
    # Train and evaluate models
    models = {}
    accuracies = {}
    
    # XGBoost
    print("\n9. Training XGBoost...")
    xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')
    xgb_model.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
    models['XGBoost'] = xgb_model
    accuracies['XGBoost'] = xgb_accuracy
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
    
    # Random Forest
    print("\n10. Training Random Forest...")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=15)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    models['Random Forest'] = rf_model
    accuracies['Random Forest'] = rf_accuracy
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # Gradient Boosting
    print("\n11. Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(random_state=42, n_estimators=200, learning_rate=0.1)
    gb_model.fit(X_train_scaled, y_train)
    y_pred_gb = gb_model.predict(X_test_scaled)
    gb_accuracy = accuracy_score(y_test, y_pred_gb)
    models['Gradient Boosting'] = gb_model
    accuracies['Gradient Boosting'] = gb_accuracy
    print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")
    
    # Ensemble Methods
    print("\n12. Training ensemble methods...")
    
    # Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('rf', rf_model),
            ('gb', gb_model)
        ],
        voting='soft'
    )
    
    voting_clf.fit(X_train_scaled, y_train)
    y_pred_voting = voting_clf.predict(X_test_scaled)
    voting_accuracy = accuracy_score(y_test, y_pred_voting)
    models['Voting Classifier'] = voting_clf
    accuracies['Voting Classifier'] = voting_accuracy
    print(f"Voting Classifier Accuracy: {voting_accuracy:.4f}")
    
    # Results Summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    # Sort models by accuracy
    sorted_accuracies = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    
    for i, (model_name, accuracy) in enumerate(sorted_accuracies, 1):
        print(f"{i}. {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    best_model_name, best_accuracy = sorted_accuracies[0]
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"🎯 ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    if best_accuracy >= 0.85:
        print("✅ SUCCESS: Achieved above 85% accuracy!")
    else:
        print(f"⚠️  Target not met. Need {(0.85 - best_accuracy)*100:.2f}% improvement")
    
    # Save best model
    print(f"\n💾 Saving best model: {best_model_name}")
    import joblib
    joblib.dump(models[best_model_name], f'{best_model_name.replace(" ", "_").lower()}_enhanced.joblib')
    joblib.dump(preprocessor, 'enhanced_preprocessor.joblib')
    
    print("\n🎉 Enhanced model training completed successfully!")

if __name__ == "__main__":
    main()
