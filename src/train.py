"""
Machine Learning Training and Evaluation Module
Student Mental Health Risk Identification and Support System

Trains, validates, compares, and exports four ML classification algorithms:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors Classifier

Focuses on Recall and F1-score to minimize false negatives (unidentified at-risk students).
Conducts 5-Fold Stratified Cross-Validation and exports models and preprocessors with Joblib.
"""

import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    from src.data_preprocessing import preprocess_student_data
except ImportError:
    from data_preprocessing import preprocess_student_data


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer for numerical and categorical pipelines."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )


def initialize_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """Initialize candidate ML models with tuned hyperparameters for high recall & generalization."""
    return {
        'Logistic Regression': LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=25,
            min_samples_leaf=15,
            class_weight='balanced',
            random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=8,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=9,
            weights='distance',
            n_jobs=-1
        )
    }


def train_and_evaluate_models(
    cleaned_csv_path: str = 'data/processed/cleaned_student_depression.csv',
    models_dir: str = 'models',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Main training and evaluation routine.
    """
    os.makedirs(models_dir, exist_ok=True)

    # Ensure processed data exists
    if not os.path.exists(cleaned_csv_path):
        print(f"Cleaned dataset not found at {cleaned_csv_path}. Running preprocessing first...")
        preprocess_student_data('data/raw/student_depression_dataset.csv', cleaned_csv_path)

    df = pd.read_csv(cleaned_csv_path)

    # Ensure engineered features exist
    if 'Perceived_Stress_Index' not in df.columns and 'Academic Pressure' in df.columns and 'Financial Stress' in df.columns:
        df['Perceived_Stress_Index'] = df['Academic Pressure'] + df['Financial Stress']
    if 'Sleep_Deficit_Flag' not in df.columns and 'Sleep Duration' in df.columns:
        df['Sleep_Deficit_Flag'] = df['Sleep Duration'].isin(['Less than 5 hours', '5-6 hours']).astype(int)

    # Define features
    num_cols = [
        'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours',
        'Financial Stress', 'Perceived_Stress_Index', 'Sleep_Deficit_Flag'
    ]
    cat_cols = ['Gender', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Family History of Mental Illness']
    target_col = 'Depression'

    # Filter features that exist
    features = num_cols + cat_cols
    X = df[features].copy()
    y = df[target_col].copy()

    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)
    models = initialize_candidate_models(random_state=random_state)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scorers = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    evaluation_report = {
        'dataset_summary': {
            'total_samples': int(len(df)),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'num_features': len(features),
            'target_distribution': {
                'class_0_low_indicators': int((y == 0).sum()),
                'class_1_elevated_indicators': int((y == 1).sum()),
                'elevated_ratio': float(round((y == 1).mean(), 4))
            }
        },
        'models': {}
    }

    best_model_name = None
    best_f1_score = -1.0
    trained_pipelines = {}

    print(f"\n=======================================================")
    print(f"Training & Evaluating ML Models on {len(df)} Students")
    print(f"Features: {features}")
    print(f"=======================================================\n")

    for name, clf in models.items():
        print(f"--> Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # 5-Fold Stratified Cross-Validation
        cv_res = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=cv_scorers, n_jobs=-1)

        # Fit on entire training set
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predict on holdout test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None

        test_acc = float(accuracy_score(y_test, y_pred))
        test_prec = float(precision_score(y_test, y_pred))
        test_rec = float(recall_score(y_test, y_pred))
        test_f1 = float(f1_score(y_test, y_pred))
        test_auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else 0.0
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Prioritizing Recall: Threshold optimization test (e.g. threshold = 0.40)
        threshold_tuned = {}
        if y_proba is not None:
            tuned_pred = (y_proba >= 0.40).astype(int)
            threshold_tuned = {
                'threshold': 0.40,
                'accuracy': float(round(accuracy_score(y_test, tuned_pred), 4)),
                'precision': float(round(precision_score(y_test, tuned_pred), 4)),
                'recall': float(round(recall_score(y_test, tuned_pred), 4)),
                'f1': float(round(f1_score(y_test, tuned_pred), 4)),
                'confusion_matrix': confusion_matrix(y_test, tuned_pred).tolist()
            }

        # Save model artifact
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        model_save_path = os.path.join(models_dir, f"{safe_name}.joblib")
        joblib.dump(pipeline, model_save_path)

        model_metrics = {
            'cross_validation': {
                'accuracy_mean': float(round(cv_res['test_accuracy'].mean(), 4)),
                'accuracy_std': float(round(cv_res['test_accuracy'].std(), 4)),
                'precision_mean': float(round(cv_res['test_precision'].mean(), 4)),
                'precision_std': float(round(cv_res['test_precision'].std(), 4)),
                'recall_mean': float(round(cv_res['test_recall'].mean(), 4)),
                'recall_std': float(round(cv_res['test_recall'].std(), 4)),
                'f1_mean': float(round(cv_res['test_f1'].mean(), 4)),
                'f1_std': float(round(cv_res['test_f1'].std(), 4)),
                'roc_auc_mean': float(round(cv_res['test_roc_auc'].mean(), 4)),
                'roc_auc_std': float(round(cv_res['test_roc_auc'].std(), 4)),
            },
            'test_set': {
                'accuracy': float(round(test_acc, 4)),
                'precision': float(round(test_prec, 4)),
                'recall': float(round(test_rec, 4)),
                'f1_score': float(round(test_f1, 4)),
                'roc_auc': float(round(test_auc, 4)),
                'confusion_matrix': cm
            },
            'threshold_tuned_0_40': threshold_tuned,
            'model_path': model_save_path
        }
        evaluation_report['models'][name] = model_metrics

        print(f"    [CV]   Acc: {cv_res['test_accuracy'].mean():.4f} | Rec: {cv_res['test_recall'].mean():.4f} | F1: {cv_res['test_f1'].mean():.4f} | AUC: {cv_res['test_roc_auc'].mean():.4f}")
        print(f"    [Test] Acc: {test_acc:.4f} | Prec: {test_prec:.4f} | Rec: {test_rec:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}")

        # Best model selection prioritized on F1 and Recall
        selection_score = (test_f1 * 0.6) + (test_rec * 0.4)
        if selection_score > best_f1_score:
            best_f1_score = selection_score
            best_model_name = name

    # Mark and save best model
    evaluation_report['best_model'] = best_model_name
    best_pipeline = trained_pipelines[best_model_name]
    best_model_path = os.path.join(models_dir, 'best_model.joblib')
    joblib.dump(best_pipeline, best_model_path)
    print(f"\n---> Best Model Selected: {best_model_name} (Saved to {best_model_path})")

    # Extract feature importance from Random Forest
    rf_pipeline = trained_pipelines['Random Forest']
    fitted_preprocessor = rf_pipeline.named_steps['preprocessor']
    fitted_rf = rf_pipeline.named_steps['classifier']

    feature_names = (
        num_cols +
        list(fitted_preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols))
    )
    importances = fitted_rf.feature_importances_
    feat_imp = [
        {'feature': feat, 'importance': float(round(imp, 5))}
        for feat, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    ]
    evaluation_report['feature_importances'] = feat_imp[:15]

    # Save feature metadata for prediction inference
    metadata = {
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'all_features': features,
        'target': target_col,
        'degree_options': sorted(df['Degree'].dropna().unique().tolist()),
        'sleep_options': ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'],
        'diet_options': ['Healthy', 'Moderate', 'Unhealthy'],
        'gender_options': ['Male', 'Female']
    }
    with open(os.path.join(models_dir, 'feature_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save metrics JSON
    with open(os.path.join(models_dir, 'model_metrics.json'), 'w') as f:
        json.dump(evaluation_report, f, indent=2)

    print(f"Metrics saved to {os.path.join(models_dir, 'model_metrics.json')}")
    print(f"Metadata saved to {os.path.join(models_dir, 'feature_metadata.json')}")

    return evaluation_report


if __name__ == '__main__':
    train_and_evaluate_models()
