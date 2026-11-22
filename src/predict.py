"""
Prediction and Inference Module
Student Mental Health Risk Identification and Support System

Provides:
1. Model loading and inference pipeline.
2. Input validation and preprocessing.
3. Ethical risk indicator score and tier calculation.
4. Key risk factors attribution.
5. Recommendation and crisis safety checks.
"""

import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.utils import map_probability_to_tier, generate_support_recommendations, CRISIS_RESOURCES
except ImportError:
    from utils import map_probability_to_tier, generate_support_recommendations, CRISIS_RESOURCES


class StudentRiskPredictor:
    """Predictor class for evaluating student mental health risk indicators."""

    def __init__(self, model_path: str = 'models/best_model.joblib', metadata_path: str = 'models/feature_metadata.json'):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = {}
        self.load_artifacts()

    def load_artifacts(self):
        """Load trained pipeline and feature metadata."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Train the model first.")
        self.model = joblib.load(self.model_path)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)

    def prepare_input(self, raw_input: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """Standardize input into DataFrame matching training schema, computing engineered features."""
        if isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        else:
            df = raw_input.copy()

        # Compute engineered features if not already provided
        if 'Perceived_Stress_Index' not in df.columns:
            acad = pd.to_numeric(df.get('Academic Pressure', 3), errors='coerce').fillna(3)
            fin = pd.to_numeric(df.get('Financial Stress', 3), errors='coerce').fillna(3)
            df['Perceived_Stress_Index'] = acad + fin

        if 'Sleep_Deficit_Flag' not in df.columns:
            sleep = df.get('Sleep Duration', pd.Series(['7-8 hours'])).astype(str)
            df['Sleep_Deficit_Flag'] = sleep.isin(['Less than 5 hours', '5-6 hours']).astype(int)

        # Coerce numeric fields
        num_cols = self.metadata.get('num_cols', [
            'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours',
            'Financial Stress', 'Perceived_Stress_Index', 'Sleep_Deficit_Flag'
        ])
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Ensure all required features are present
        all_features = self.metadata.get('all_features', [
            'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours',
            'Financial Stress', 'Perceived_Stress_Index', 'Sleep_Deficit_Flag',
            'Gender', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Family History of Mental Illness'
        ])
        for col in all_features:
            if col not in df.columns:
                df[col] = np.nan

        return df[all_features]

    def identify_key_risk_factors(self, student_dict: Dict[str, Any]) -> list:
        """Identify key lifestyle and academic stressors contributing to elevated indicators."""
        factors = []

        acad_p = float(student_dict.get('Academic Pressure', 0))
        if acad_p >= 4:
            factors.append({
                "factor": "High Academic Pressure",
                "value": f"{int(acad_p)} / 5",
                "impact": "High",
                "explanation": "Severe perceived academic pressure is strongly correlated with emotional exhaustion and academic burnout."
            })

        sleep = str(student_dict.get('Sleep Duration', ''))
        if sleep in ['Less than 5 hours', '5-6 hours']:
            factors.append({
                "factor": "Sleep Deprivation",
                "value": sleep,
                "impact": "High",
                "explanation": "Getting under 6 hours of sleep impairs cognitive resilience and emotional regulation."
            })

        fin_s = float(student_dict.get('Financial Stress', 0))
        if fin_s >= 4:
            factors.append({
                "factor": "Elevated Financial Stress",
                "value": f"{int(fin_s)} / 5",
                "impact": "High",
                "explanation": "Financial burden acts as a continuous background stressor affecting academic focus."
            })

        study_hrs = float(student_dict.get('Work/Study Hours', 0))
        if study_hrs >= 9:
            factors.append({
                "factor": "Heavy Daily Study Load",
                "value": f"{study_hrs:.1f} hrs/day",
                "impact": "Moderate",
                "explanation": "Long unbroken study hours without restorative breaks reduce focus and increase burnout risk."
            })

        study_sat = float(student_dict.get('Study Satisfaction', 5))
        if study_sat <= 2:
            factors.append({
                "factor": "Low Study Satisfaction",
                "value": f"{int(study_sat)} / 5",
                "impact": "Moderate",
                "explanation": "Low satisfaction with current studies can lead to disengagement and feelings of helplessness."
            })

        diet = str(student_dict.get('Dietary Habits', ''))
        if diet == 'Unhealthy':
            factors.append({
                "factor": "Unhealthy Nutrition Patterns",
                "value": diet,
                "impact": "Moderate",
                "explanation": "Irregular nutrition negatively affects energy levels, circadian rhythm, and stress tolerance."
            })

        fam_hist = str(student_dict.get('Family History of Mental Illness', 'No'))
        if fam_hist == 'Yes':
            factors.append({
                "factor": "Family History of Mental Health Challenges",
                "value": "Yes",
                "impact": "Moderate",
                "explanation": "Genetic or familial factors can increase susceptibility during periods of high academic strain."
            })

        return factors

    def predict_single(self, student_dict: Dict[str, Any], custom_threshold: float = None) -> Dict[str, Any]:
        """
        Run inference on a single student survey response.
        """
        input_df = self.prepare_input(student_dict)

        # Get probability
        if hasattr(self.model, 'predict_proba'):
            prob = float(self.model.predict_proba(input_df)[0, 1])
        else:
            pred = int(self.model.predict(input_df)[0])
            prob = 1.0 if pred == 1 else 0.0

        # Map to tier
        tier_info = map_probability_to_tier(prob)

        # Custom threshold check if provided
        if custom_threshold is not None:
            tier_info['custom_threshold'] = custom_threshold
            tier_info['flagged_at_threshold'] = bool(prob >= custom_threshold)

        # Check for immediate crisis flag (e.g. suicidal ideation self-report)
        suicidal_thoughts = str(student_dict.get('Have you ever had suicidal thoughts ?', 'No')).strip().lower()
        has_crisis_flag = suicidal_thoughts in ['yes', 'true', '1']

        # Key risk factors
        key_factors = self.identify_key_risk_factors(student_dict)

        # Support recommendations
        recommendations = generate_support_recommendations(student_dict, tier_info['tier'])

        return {
            'risk_assessment': tier_info,
            'crisis_flag': has_crisis_flag,
            'crisis_resources': CRISIS_RESOURCES if has_crisis_flag else None,
            'key_contributing_factors': key_factors,
            'recommendations': recommendations,
            'model_used': type(self.model.named_steps['classifier']).__name__,
            'ethical_disclaimer': (
                "Educational Research and Early-Support Screening Tool Only. "
                "This assessment does NOT constitute a medical or clinical diagnosis. "
                "Always consult a qualified healthcare or counseling professional for clinical support."
            )
        }


if __name__ == '__main__':
    predictor = StudentRiskPredictor()
    sample_student = {
        'Age': 21,
        'Gender': 'Female',
        'Degree': 'B.Tech',
        'CGPA': 7.8,
        'Academic Pressure': 5,
        'Study Satisfaction': 2,
        'Sleep Duration': 'Less than 5 hours',
        'Dietary Habits': 'Unhealthy',
        'Work/Study Hours': 10,
        'Financial Stress': 4,
        'Family History of Mental Illness': 'Yes',
        'Have you ever had suicidal thoughts ?': 'No'
    }
    result = predictor.predict_single(sample_student)
    print("=== Sample Student Assessment ===")
    print(f"Risk Tier: {result['risk_assessment']['tier']} ({result['risk_assessment']['percentage']})")
    print(f"Key Factors: {[f['factor'] for f in result['key_contributing_factors']]}")
    print(f"Total Recommendations: {len(result['recommendations'])}")
    print(f"Disclaimer: {result['ethical_disclaimer']}")
