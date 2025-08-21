import warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import joblib

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

DATA_PATH = Path("1d9dc774-4b9b-4fb5-bc9b-a72ea185b69e.csv")  # adjust if needed

TARGET = "CVD Risk Level"

df = pd.read_csv(DATA_PATH)
y = df[TARGET].astype(str).str.strip()
X = df.drop(columns=[TARGET])

# Encode target labels for XGBoost compatibility
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print('Target distribution:', y.value_counts())
print('Encoded target distribution:', pd.Series(y_encoded).value_counts())
print('Label mapping:', dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Use encoded labels for training
y = y_encoded

cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

from sklearn import __version__ as skver
sk_major, sk_minor = [int(x) for x in skver.split('.')[:2]]
ohe_kwargs = {'handle_unknown': 'ignore'}
if (sk_major, sk_minor) >= (1, 2):
    ohe_kwargs['sparse_output'] = False
else:
    ohe_kwargs['sparse'] = False

numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(**ohe_kwargs))])

preprocess = ColumnTransformer([('num', numeric_transformer, num_cols), ('cat', categorical_transformer, cat_cols)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# Optional: XGBoost if available
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

candidates = {
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight='balanced_subsample'),
}

# Add XGBoost if available
if XGB_AVAILABLE:
    candidates["XGBoost"] = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        max_depth=6,
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE
    )

rows = []
for name, clf in candidates.items():
    pipe = Pipeline([('prep', preprocess), ('clf', clf)])
    scores = cross_val_score(pipe, X_train, y_train, scoring='accuracy', cv=5, n_jobs=-1)
    rows.append((name, scores.mean()))
rows.sort(key=lambda t: t[1], reverse=True)
top = rows[0][0]

if top == "HistGradientBoosting":
    base = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    param_distributions = {
        'clf__max_depth': [None, 3, 5, 7],
        'clf__learning_rate': [0.05, 0.1, 0.2],
        'clf__max_leaf_nodes': [15, 31, 63],
        'clf__min_samples_leaf': [10, 20, 30],
    }
elif top == "XGBoost" and XGB_AVAILABLE:
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
    base = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced_subsample')
    param_distributions = {
        'clf__n_estimators': [200, 300, 400, 600],
        'clf__max_depth': [None, 8, 12, 16],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['auto', 'sqrt', 0.5],
    }

pipe = Pipeline([('prep', preprocess), ('clf', base)])

search = RandomizedSearchCV(pipe, param_distributions=param_distributions, n_iter=25, scoring='accuracy', n_jobs=-1, cv=5, random_state=RANDOM_STATE, verbose=1)
search.fit(X_train, y_train)

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Convert predictions back to original labels for interpretability
y_pred_original = label_encoder.inverse_transform(y_pred)
y_test_original = label_encoder.inverse_transform(y_test)

print("Best CV accuracy:", search.best_score_)
print("Test accuracy:", acc)
print("\nClassification Report (Original Labels):")
print(classification_report(y_test_original, y_pred_original))
print("\nLabel mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

joblib.dump(best_model, "cvd_best_model.joblib")
joblib.dump(label_encoder, "cvd_label_encoder.joblib")
print("Saved model to cvd_best_model.joblib")
print("Saved label encoder to cvd_label_encoder.joblib")
