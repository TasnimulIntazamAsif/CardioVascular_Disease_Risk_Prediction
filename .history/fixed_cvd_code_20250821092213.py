# Fixed CVD Risk Classification Code
# This code fixes the XGBoost compatibility issue by adding label encoding

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Optional: XGBoost if available
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
    print("XGBoost is available")
    
except Exception:
    XGB_AVAILABLE = False
    print("XGBoost is not available")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load data
df = pd.read_csv('CVD_Dataset.csv')

# Define target
TARGET = 'CVD Risk Level'
assert TARGET in df.columns, f"Target column '{TARGET}' not found. Columns: {df.columns.tolist()}"

y = df[TARGET].astype(str).str.strip()
X = df.drop(columns=[TARGET])

# FIX: Encode target labels for XGBoost compatibility
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print('Target distribution:\n', y.value_counts())
print('\nEncoded target distribution:\n', pd.Series(y_encoded).value_counts())
print('\nLabel mapping:', dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
print('\nColumns by dtype:')
print(df.dtypes.value_counts())

# Use encoded labels for training
y = y_encoded

# Preprocessing
from sklearn import __version__ as skver
sk_major, sk_minor = [int(x) for x in skver.split('.')[:2]]
ohe_kwargs = {'handle_unknown': 'ignore'}
if (sk_major, sk_minor) >= (1, 2):
    ohe_kwargs['sparse_output'] = False
else:
    ohe_kwargs['sparse'] = False

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(**ohe_kwargs))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print(f'\nData shapes: X_train: {X_train.shape}, X_test: {X_test.shape}')
print(f'Number of numeric columns: {len(num_cols)}, categorical columns: {len(cat_cols)}')

# Quick Baselines
candidates = {
    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    'RandomForest': RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight='balanced_subsample'),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=400, random_state=RANDOM_STATE, class_weight='balanced'),
    'LogisticRegression': LogisticRegression(max_iter=2000, class_weight='balanced'),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

rows = []
for name, clf in candidates.items():
    pipe = Pipeline(steps=[('prep', preprocess), ('clf', clf)])
    scores = cross_val_score(pipe, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    rows.append({'model': name, 'cv_mean_acc': scores.mean(), 'cv_std': scores.std(), 'cv_scores': scores})
    
# FIX: Now XGBoost will work with encoded labels
if XGB_AVAILABLE:
    xgb = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        max_depth=6,
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE
    )
    pipe = Pipeline(steps=[('prep', preprocess), ('clf', xgb)])
    scores = cross_val_score(pipe, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    rows.append({'model': 'XGBoost', 'cv_mean_acc': scores.mean(), 'cv_std': scores.std(), 'cv_scores': scores})

baseline_df = pd.DataFrame(rows).sort_values('cv_mean_acc', ascending=False).reset_index(drop=True)
print("\nBaseline Results:")
print(baseline_df)

# Hyperparameter Tuning (Top Model)
top_name = baseline_df.iloc[0]['model']
print(f'\nTop baseline model: {top_name}')

if top_name == 'HistGradientBoosting':
    base = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    param_distributions = {
        'clf__max_depth': [None, 3, 5, 7],
        'clf__learning_rate': [0.05, 0.1, 0.2],
        'clf__max_leaf_nodes': [15, 31, 63],
        'clf__min_samples_leaf': [10, 20, 30],
    }
elif top_name == 'RandomForest':
    base = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced_subsample')
    param_distributions = {
        'clf__n_estimators': [200, 300, 400, 600],
        'clf__max_depth': [None, 8, 12, 16],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['auto', 'sqrt', 0.5],
    }
elif top_name == 'ExtraTrees':
    base = ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight='balanced')
    param_distributions = {
        'clf__n_estimators': [300, 400, 600, 800],
        'clf__max_depth': [None, 8, 12, 16],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['auto', 'sqrt', 0.5],
    }
elif top_name == 'LogisticRegression':
    base = LogisticRegression(max_iter=4000, class_weight='balanced')
    param_distributions = {
        'clf__C': np.logspace(-2, 2, 10),
        'clf__penalty': ['l2'],
        'clf__solver': ['lbfgs', 'liblinear', 'saga'],
    }
elif top_name == 'XGBoost' and XGB_AVAILABLE:
    base = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        tree_method='hist'
    )
    param_distributions = {
        'clf__max_depth': [4, 6, 8],
        'clf__n_estimators': [300, 500, 800],
        'clf__learning_rate': [0.03, 0.05, 0.1],
        'clf__subsample': [0.8, 1.0],
        'clf__colsample_bytree': [0.7, 0.9, 1.0],
    }
else:
    # Fallback to HistGradientBoosting if XGBoost wasn't available
    top_name = 'HistGradientBoosting'
    base = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    param_distributions = {
        'clf__max_depth': [None, 3, 5, 7],
        'clf__learning_rate': [0.05, 0.1, 0.2],
        'clf__max_leaf_nodes': [15, 31, 63],
        'clf__min_samples_leaf': [10, 20, 30],
    }

pipe = Pipeline(steps=[('prep', preprocess), ('clf', base)])

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=30,
    scoring='accuracy',
    n_jobs=-1,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    verbose=1,
    random_state=RANDOM_STATE
)
search.fit(X_train, y_train)

print('Best CV accuracy:', search.best_score_)
print('Best params:', search.best_params_)

best_model = search.best_estimator_

# Final Evaluation on Holdout Test Set
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Convert predictions back to original labels for interpretability
y_pred_original = label_encoder.inverse_transform(y_pred)
y_test_original = label_encoder.inverse_transform(y_test)

print(f'\nTest Accuracy: {acc:.4f}')
print('\nClassification Report (Original Labels):')
print(classification_report(y_test_original, y_pred_original))

# Save the model and label encoder
import joblib
joblib.dump(best_model, "cvd_best_model.joblib")
joblib.dump(label_encoder, "cvd_label_encoder.joblib")
print('\nSaved model to cvd_best_model.joblib')
print('Saved label encoder to cvd_label_encoder.joblib')

print('\nSUCCESS! The XGBoost compatibility issue has been fixed.')
print('The target labels are now properly encoded as numbers:')
print('Label mapping:', dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
