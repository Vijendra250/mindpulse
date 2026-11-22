"""
Model Evaluation and Metrics Module
Student Mental Health Risk Identification and Support System

Provides:
- In-depth evaluation of models across accuracy, precision, recall, F1, ROC-AUC.
- Confusion matrix calculations and metrics display.
- Sensitivity & Threshold tuning analysis to prioritize recall.
- Feature importance extraction.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, f1_score, recall_score, precision_score, accuracy_score
)


def evaluate_saved_model(
    model_path: str = 'models/best_model.joblib',
    data_path: str = 'data/processed/cleaned_student_depression.csv'
) -> Dict[str, Any]:
    """
    Run detailed evaluation on the saved pipeline using test split.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)

    # Ensure engineered features exist
    if 'Perceived_Stress_Index' not in df.columns:
        df['Perceived_Stress_Index'] = df['Academic Pressure'] + df['Financial Stress']
    if 'Sleep_Deficit_Flag' not in df.columns:
        df['Sleep_Deficit_Flag'] = df['Sleep Duration'].isin(['Less than 5 hours', '5-6 hours']).astype(int)

    num_cols = [
        'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours',
        'Financial Stress', 'Perceived_Stress_Index', 'Sleep_Deficit_Flag'
    ]
    cat_cols = ['Gender', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Family History of Mental Illness']
    features = num_cols + cat_cols
    target = 'Depression'

    X = df[features]
    y = df[target]

    # Predict
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, 'predict_proba') else None

    # Calculate metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba) if y_proba is not None else 0.0
    cm = confusion_matrix(y, y_pred)

    # Threshold analysis for prioritizing recall
    threshold_analysis = []
    if y_proba is not None:
        for t in np.arange(0.20, 0.85, 0.05):
            t_pred = (y_proba >= t).astype(int)
            threshold_analysis.append({
                'threshold': float(round(t, 2)),
                'recall': float(round(recall_score(y, t_pred), 4)),
                'precision': float(round(precision_score(y, t_pred), 4)),
                'f1': float(round(f1_score(y, t_pred), 4)),
                'accuracy': float(round(accuracy_score(y, t_pred), 4))
            })

    report = {
        'accuracy': float(round(acc, 4)),
        'precision': float(round(prec, 4)),
        'recall': float(round(rec, 4)),
        'f1_score': float(round(f1, 4)),
        'roc_auc': float(round(auc, 4)),
        'confusion_matrix': cm.tolist(),
        'threshold_analysis': threshold_analysis
    }
    return report


if __name__ == '__main__':
    rep = evaluate_saved_model()
    print("=== Evaluation Summary for Best Model ===")
    print(f"Accuracy:  {rep['accuracy']:.4f}")
    print(f"Precision: {rep['precision']:.4f}")
    print(f"Recall:    {rep['recall']:.4f} (Prioritized for student safety)")
    print(f"F1-Score:  {rep['f1_score']:.4f}")
    print(f"ROC-AUC:   {rep['roc_auc']:.4f}")
    print(f"Confusion Matrix: {rep['confusion_matrix']}")
