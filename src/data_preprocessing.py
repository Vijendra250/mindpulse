"""
Data Preprocessing Module
Student Mental Health Risk Identification and Support System

This module handles:
1. Loading raw student survey data from Kaggle.
2. Cleaning, trimming whitespace, and stripping quotation artifacts.
3. Filtering to student population (removing non-student records).
4. Dropping non-predictive identifiers and data leakage candidates (id, city, work pressure).
5. Coercing and imputing numerical and categorical features.
6. Identifying outliers via IQR analysis.
7. Feature engineering of composite indices (e.g., Perceived Stress Index).
8. Exporting clean processed datasets.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip extraneous quotation marks and whitespace from string columns."""
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=['object', 'str']).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip("'\" \t\r\n")
    return df_clean


def detect_outliers_iqr(df: pd.DataFrame, columns: list) -> Dict[str, Dict[str, Any]]:
    """
    Detect outliers in numerical columns using the Interquartile Range (IQR) method.
    Returns outlier summary per column.
    """
    outlier_summary = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        q25 = np.percentile(series, 25)
        q75 = np.percentile(series, 75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_summary[col] = {
            'q25': float(round(q25, 2)),
            'q75': float(round(q75, 2)),
            'iqr': float(round(iqr, 2)),
            'lower_bound': float(round(lower_bound, 2)),
            'upper_bound': float(round(upper_bound, 2)),
            'outlier_count': int(len(outliers)),
            'outlier_percentage': float(round(len(outliers) / len(series) * 100, 2)),
            'min': float(round(series.min(), 2)),
            'max': float(round(series.max(), 2))
        }
    return outlier_summary


def preprocess_student_data(raw_csv_path: str, output_csv_path: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main preprocessing pipeline for the Student Mental Health / Depression dataset.

    Parameters:
    - raw_csv_path: Path to raw CSV file.
    - output_csv_path: Optional path to save cleaned CSV.

    Returns:
    - cleaned_df: Cleaned pandas DataFrame ready for ML and EDA.
    - metadata: Dictionary containing data cleaning report and outlier statistics.
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_csv_path}")

    # Step 1: Load raw data
    raw_df = pd.read_csv(raw_csv_path)
    initial_rows, initial_cols = raw_df.shape
    metadata = {
        'initial_shape': (initial_rows, initial_cols),
        'dropped_columns': [],
        'non_student_rows_dropped': 0,
        'duplicate_rows_dropped': 0,
        'missing_values_handled': {},
    }

    # Step 2: Clean strings
    df = clean_string_columns(raw_df)

    # Step 3: Filter for students only
    if 'Profession' in df.columns:
        student_mask = df['Profession'] == 'Student'
        non_student_count = int((~student_mask).sum())
        metadata['non_student_rows_dropped'] = non_student_count
        df = df[student_mask].copy()

    # Step 4: Drop redundant, non-student, and identifier columns to prevent data leakage
    leakage_and_id_cols = ['id', 'Profession', 'Work Pressure', 'Job Satisfaction', 'City']
    cols_to_drop = [c for c in leakage_and_id_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    metadata['dropped_columns'] = cols_to_drop

    # Step 5: Handle non-standard missing values (e.g., '?' in Financial Stress)
    if 'Financial Stress' in df.columns:
        df['Financial Stress'] = pd.to_numeric(df['Financial Stress'].replace('?', np.nan), errors='coerce')
        missing_fin = int(df['Financial Stress'].isna().sum())
        if missing_fin > 0:
            median_fin = df['Financial Stress'].median()
            df['Financial Stress'] = df['Financial Stress'].fillna(median_fin)
            metadata['missing_values_handled']['Financial Stress'] = {
                'missing_count': missing_fin,
                'imputation_strategy': 'median',
                'imputed_value': float(median_fin)
            }

    # Step 6: Standardize numeric column types
    numeric_columns = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours', 'Financial Stress']
    for num_col in numeric_columns:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors='coerce')
            missing_num = int(df[num_col].isna().sum())
            if missing_num > 0:
                med_val = df[num_col].median()
                df[num_col] = df[num_col].fillna(med_val)
                metadata['missing_values_handled'][num_col] = {
                    'missing_count': missing_num,
                    'imputation_strategy': 'median',
                    'imputed_value': float(med_val)
                }

    # Step 7: Standardize categorical variables
    categorical_columns = ['Gender', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
    for cat_col in categorical_columns:
        if cat_col in df.columns:
            # Replace invalid placeholders
            df[cat_col] = df[cat_col].replace({'?': 'Others', 'nan': 'Others', 'None': 'Others'})
            if cat_col == 'Sleep Duration':
                # Standardize common sleep duration bins
                valid_sleep = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
                df[cat_col] = df[cat_col].apply(lambda x: x if x in valid_sleep else 'Others')
            elif cat_col == 'Dietary Habits':
                valid_diet = ['Healthy', 'Moderate', 'Unhealthy']
                df[cat_col] = df[cat_col].apply(lambda x: x if x in valid_diet else 'Others')

    # Step 8: Standardize Target variable (Depression)
    if 'Depression' in df.columns:
        df['Depression'] = pd.to_numeric(df['Depression'], errors='coerce').fillna(0).astype(int)

    # Step 9: Remove duplicate records
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        df = df.drop_duplicates()
    metadata['duplicate_rows_dropped'] = dup_count

    # Step 10: Feature engineering - domain specific risk indicators
    # Perceived Stress Index (Sum of Academic Pressure 1-5 and Financial Stress 1-5 => Scale 2 to 10)
    if 'Academic Pressure' in df.columns and 'Financial Stress' in df.columns:
        df['Perceived_Stress_Index'] = df['Academic Pressure'] + df['Financial Stress']

    # Sleep Deficit Flag: True if sleeping Less than 5 hours or 5-6 hours
    if 'Sleep Duration' in df.columns:
        df['Sleep_Deficit_Flag'] = df['Sleep Duration'].isin(['Less than 5 hours', '5-6 hours']).astype(int)

    # Step 11: Outlier detection
    metadata['outlier_analysis'] = detect_outliers_iqr(df, numeric_columns)
    metadata['final_shape'] = (int(df.shape[0]), int(df.shape[1]))
    metadata['final_columns'] = df.columns.tolist()

    # Step 12: Save cleaned dataset if requested
    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"Cleaned dataset saved successfully to: {output_csv_path}")

    return df, metadata


if __name__ == '__main__':
    raw_path = 'data/raw/student_depression_dataset.csv'
    clean_path = 'data/processed/cleaned_student_depression.csv'
    print("Executing data preprocessing pipeline...")
    cleaned_df, report = preprocess_student_data(raw_path, clean_path)
    print("\n--- Preprocessing Summary ---")
    print(f"Initial Shape: {report['initial_shape']}")
    print(f"Non-student rows dropped: {report['non_student_rows_dropped']}")
    print(f"Columns dropped: {report['dropped_columns']}")
    print(f"Missing values handled: {report['missing_values_handled']}")
    print(f"Final Shape: {report['final_shape']}")
    print("\nOutlier Detection Highlights:")
    for col, stats in report['outlier_analysis'].items():
        print(f"  {col}: {stats['outlier_count']} outliers ({stats['outlier_percentage']}%) [Bounds: {stats['lower_bound']} to {stats['upper_bound']}]")
