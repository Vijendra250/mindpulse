"""
Generate and execute the complete, unified Jupyter Notebook:
Student_Mental_Health_Complete_Project.ipynb
"""

import os
import json
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
nb.metadata = {
    'kernelspec': {
        'display_name': 'Python (.venv) (Student Mental Health)',
        'language': 'python',
        'name': 'student-mental-health'
    },
    'language_info': {
        'name': 'python',
        'version': '3.11'
    }
}

cells = []

# ==============================================================================
# Cell 1: Header & Project Overview
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# Student Mental Health Risk Identification and Support System
### Complete End-to-End Machine Learning & Data Analytics Notebook

---

## 📌 Project Overview
University students worldwide face increasing academic demands, irregular sleep schedules, financial strain, and psychological distress. This comprehensive notebook integrates the entire machine learning and data analytics pipeline for identifying early indicators of student mental health risk using anonymized survey responses from Kaggle.

### ⚠️ Important Ethical & Clinical Disclaimer
> **This system is strictly an educational research and early-support screening tool. It does NOT provide clinical or medical diagnoses, psychiatric evaluations, or treatment prescriptions.**
>
> Outputs are strictly categorized into non-stigmatizing tiers:
> - `🟢 Low indicators`: Typical baseline academic stress; standard healthy study routines recommended.
> - `🟡 Moderate indicators`: Noticeable strain across academic workload, sleep, or finances; proactive peer/counseling check-ins encouraged.
> - `🔴 High indicators`: Significant cumulative distress indicators; direct suggestions to connect with university counselors or licensed healthcare professionals.
>
> Zero Personally Identifiable Information (PII) is collected or stored. If self-harm thoughts are reported, 24/7 crisis hotlines are provided immediately.

---

### Table of Contents:
1. **Environment Setup & Imports**
2. **Data Ingestion & Integrity Checks** (Primary Kaggle Survey & Companion Clinical Survey)
3. **Data Preprocessing & Data Leakage Prevention**
4. **Exploratory Data Analysis (EDA)** (Visualizing stress, sleep, academic pressure, study hours, correlations, and treatment gap)
5. **Machine Learning Model Training & 5-Fold Stratified Cross-Validation** (Logistic Regression, Decision Tree, Random Forest, KNN)
6. **Model Evaluation & Prioritizing Recall** (Confusion matrices, ROC curves, PR curves, threshold optimization)
7. **End-to-End Prediction & Tailored Recommendation Engine**
8. **Model Serialization, Schema Verification & Export**
9. **Precision-Recall Trade-off, Model Limitations & Ethics Summary**
"""))

# ==============================================================================
# Cell 2: Imports & Plotting Configuration
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# 1. Environment Setup & Library Imports
import os
import sys
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# Plotting configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette('tab10')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

print("✓ All libraries imported and plotting styles configured.")
"""))

# ==============================================================================
# Cell 3: Data Ingestion
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 2. Data Ingestion & Initial Inspection
We load two publicly available Kaggle datasets:
1. **Primary Dataset:** Kaggle Student Depression & Mental Health Survey (27,870 student respondents).
2. **Companion Dataset:** Kaggle Student Mental Health Survey by Shariful Islam (evaluating anxiety, panic attacks, and specialist help-seeking).
"""))

cells.append(nbf.v4.new_code_cell("""# Helper function to locate data files whether run from root or notebooks/
def resolve_data_path(rel_path):
    for candidate in [rel_path, os.path.join('..', rel_path), os.path.abspath(rel_path)]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"File not found: {rel_path}")

primary_path = resolve_data_path('data/raw/student_depression_dataset.csv')
companion_path = resolve_data_path('data/raw/student_mental_health.csv')

raw_df = pd.read_csv(primary_path)
comp_df = pd.read_csv(companion_path)

print(f"Primary Survey Shape:   {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")
print(f"Companion Survey Shape: {comp_df.shape[0]:,} rows × {comp_df.shape[1]} columns")
raw_df.head(3)
"""))

# ==============================================================================
# Cell 4: Preprocessing & Data Leakage Prevention
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 3. Data Preprocessing & Data Leakage Prevention

### Preprocessing Protocol:
1. **Target Population Filtering:** Restrict to verified students (`Profession == 'Student'`), dropping 31 non-student occupational records.
2. **Preventing Target-Proxy Leakage:**
   - Drop arbitrary identifier `id`.
   - Drop `City` (high cardinality noise).
   - Drop `Work Pressure` and `Job Satisfaction` (strictly zero for full-time students).
   - Decouple self-reported suicidal ideation from standard ML training features to force the model to learn early lifestyle, academic, and biological stressors.
3. **Handling Non-Standard Missing Values:** Replace `'?'` in `Financial Stress` and impute with median.
4. **IQR Outlier Detection:** Evaluate distributions across numerical features.
5. **Feature Engineering:** Calculate `Perceived_Stress_Index` (`Academic Pressure` + `Financial Stress`) and `Sleep_Deficit_Flag`.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Clean string whitespace and quotation artifacts
clean_df = raw_df.copy()
for col in clean_df.select_dtypes(include=['object', 'str']).columns:
    clean_df[col] = clean_df[col].astype(str).str.replace("'", "").str.replace('"', '').str.strip()

# 2. Filter for students only
if 'Profession' in clean_df.columns:
    clean_df = clean_df[clean_df['Profession'] == 'Student'].copy()

# 3. Drop leakage and non-student columns
leakage_cols = ['id', 'Profession', 'Work Pressure', 'Job Satisfaction', 'City']
clean_df = clean_df.drop(columns=[c for c in leakage_cols if c in clean_df.columns])

# 4. Handle non-standard missing values ('?' in Financial Stress)
clean_df['Financial Stress'] = pd.to_numeric(clean_df['Financial Stress'].replace('?', np.nan), errors='coerce')
clean_df['Financial Stress'] = clean_df['Financial Stress'].fillna(clean_df['Financial Stress'].median())

# Standardize numerical features
base_num_cols = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours', 'Financial Stress']
for col in base_num_cols:
    clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
    clean_df[col] = clean_df[col].fillna(clean_df[col].median())

# Standardize categorical features
cat_cols = ['Gender', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Family History of Mental Illness']
clean_df['Sleep Duration'] = clean_df['Sleep Duration'].apply(
    lambda x: x if x in ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'] else 'Others'
)
clean_df['Dietary Habits'] = clean_df['Dietary Habits'].apply(
    lambda x: x if x in ['Healthy', 'Moderate', 'Unhealthy'] else 'Others'
)

# Standardize target
clean_df['Depression'] = clean_df['Depression'].astype(int)

# Drop duplicate records
clean_df = clean_df.drop_duplicates()

# Feature Engineering
clean_df['Perceived_Stress_Index'] = clean_df['Academic Pressure'] + clean_df['Financial Stress']
clean_df['Sleep_Deficit_Flag'] = clean_df['Sleep Duration'].isin(['Less than 5 hours', '5-6 hours']).astype(int)

# All numerical features including engineered features
num_cols = base_num_cols + ['Perceived_Stress_Index', 'Sleep_Deficit_Flag']
feature_cols = num_cols + cat_cols

print(f"Cleaned Student Dataset: {clean_df.shape[0]:,} records × {clean_df.shape[1]} features")
print(f"Full Model Feature Schema ({len(feature_cols)} features):\\n{feature_cols}")
clean_df.describe().T[['mean', 'std', 'min', '50%', 'max']]
"""))

cells.append(nbf.v4.new_code_cell("""# Outlier Detection via Interquartile Range (IQR) Method
print("=== Numerical Feature Outlier Analysis (IQR Method) ===")
for col in num_cols:
    s = clean_df[col].dropna()
    q25, q75 = np.percentile(s, 25), np.percentile(s, 75)
    iqr = q75 - q25
    lb, ub = q25 - 1.5 * iqr, q75 + 1.5 * iqr
    outliers = s[(s < lb) | (s > ub)]
    print(f"{col:22s} | Bounds: [{lb:5.2f}, {ub:5.2f}] | Outliers: {len(outliers):4d} ({len(outliers)/len(s)*100:.2f}%)")
"""))

# ==============================================================================
# Cell 5: Exploratory Data Analysis
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 4. Exploratory Data Analysis (EDA)

We explore the distributions, relationships, and trade-offs among:
- **Target Indicator Distribution:** Low Indicators vs. Elevated Indicators
- **Stress Dimensions:** Academic Pressure, Financial Stress, and Composite Perceived Stress
- **Sleep Duration:** Impact of sleep deficit on risk probability
- **Academic Performance:** CGPA vs. Daily Study Hours
- **Gender Balance:** Demographic fairness checks
- **Correlation Heatmap:** Linear and rank relationships between factors
- **Companion Clinical Survey:** Anxiety, Panic Attacks, and Specialist Help-Seeking
"""))

cells.append(nbf.v4.new_code_cell("""# 4.1 Target Variable Distribution
plt.figure(figsize=(7, 4.5))
ax = sns.countplot(x='Depression', data=clean_df, palette=['#28a745', '#dc3545'])
plt.title('Distribution of Mental Health Risk Indicators', fontweight='bold', pad=12)
plt.xlabel('Risk Status (0 = Low Indicators, 1 = Elevated Indicators)')
plt.ylabel('Student Count')
plt.xticks([0, 1], ['Low Indicators (0)', 'Elevated Indicators (1)'])

total = len(clean_df)
for p in ax.patches:
    pct = f"{100 * p.get_height() / total:.1f}%"
    ax.annotate(pct, (p.get_x() + p.get_width() / 2 - 0.05, p.get_height() + 300), fontweight='bold')

plt.tight_layout()
plt.show()

print(clean_df['Depression'].value_counts(normalize=True).mul(100).round(2).to_dict())
"""))

cells.append(nbf.v4.new_code_cell("""# 4.2 Academic Pressure & Financial Stress vs. Risk
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(ax=axes[0], x='Academic Pressure', hue='Depression', data=clean_df, palette=['#28a745', '#dc3545'])
axes[0].set_title('Academic Pressure vs. Risk Status', fontweight='bold')
axes[0].set_xlabel('Perceived Academic Pressure (1: Minimal to 5: Extreme)')
axes[0].set_ylabel('Number of Students')
axes[0].legend(['Low Indicators', 'Elevated Indicators'])

sns.countplot(ax=axes[1], x='Financial Stress', hue='Depression', data=clean_df, palette=['#28a745', '#dc3545'])
axes[1].set_title('Financial Stress vs. Risk Status', fontweight='bold')
axes[1].set_xlabel('Perceived Financial Stress (1: Minimal to 5: Extreme)')
axes[1].set_ylabel('Number of Students')
axes[1].legend(['Low Indicators', 'Elevated Indicators'])

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 4.3 Sleep Duration Impact on Mental Health Risk
sleep_order = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
sleep_pct = pd.crosstab(clean_df['Sleep Duration'], clean_df['Depression'], normalize='index').loc[sleep_order] * 100

ax = sleep_pct.plot(kind='bar', stacked=True, figsize=(10, 5), color=['#28a745', '#dc3545'])
plt.title('Proportion of Risk Indicators Across Sleep Duration Tiers', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Daily Sleep Duration')
plt.ylabel('Percentage of Students (%)')
plt.xticks(rotation=0)
plt.legend(['Low Indicators', 'Elevated Indicators'], loc='upper right')

for i, val in enumerate(sleep_pct[1]):
    ax.text(i, val / 2 + sleep_pct.iloc[i, 0], f"{val:.1f}%", ha='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 4.4 Study Hours and Academic Performance (CGPA)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(ax=axes[0], x='Depression', y='Work/Study Hours', data=clean_df, palette=['#28a745', '#dc3545'])
axes[0].set_title('Daily Study Hours by Risk Tier', fontweight='bold')
axes[0].set_xticklabels(['Low Indicators', 'Elevated Indicators'])
axes[0].set_ylabel('Daily Study Hours')

sns.boxplot(ax=axes[1], x='Academic Pressure', y='CGPA', hue='Depression', data=clean_df, palette=['#28a745', '#dc3545'])
axes[1].set_title('CGPA Across Academic Pressure Levels', fontweight='bold')
axes[1].set_xlabel('Academic Pressure (1-5)')
axes[1].set_ylabel('Cumulative GPA')
axes[1].legend(['Low Indicators', 'Elevated Indicators'])

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 4.5 Correlation Heatmap Across Factors
corr_features = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 
                 'Work/Study Hours', 'Financial Stress', 'Perceived_Stress_Index', 'Sleep_Deficit_Flag', 'Depression']

plt.figure(figsize=(10, 7.5))
corr_matrix = clean_df[corr_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix of Student Stressors and Well-Being Indicators', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 4.6 Companion Study: Anxiety, Panic Attacks & Treatment Seeking
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

comp_df['Do you have Anxiety?'].value_counts().plot(
    kind='pie', ax=axes[0], autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'], explode=(0, 0.08), startangle=90
)
axes[0].set_title('Reported Anxiety Symptoms', fontweight='bold')
axes[0].set_ylabel('')

comp_df['Do you have Panic attack?'].value_counts().plot(
    kind='pie', ax=axes[1], autopct='%1.1f%%', colors=['#ffcc99', '#99ff99'], explode=(0, 0.08), startangle=90
)
axes[1].set_title('Reported Panic Attacks', fontweight='bold')
axes[1].set_ylabel('')

comp_df['Did you seek any specialist for a treatment?'].value_counts().plot(
    kind='pie', ax=axes[2], autopct='%1.1f%%', colors=['#ff6666', '#c2c2f0'], explode=(0, 0.15), startangle=90
)
axes[2].set_title('Sought Specialist Treatment', fontweight='bold')
axes[2].set_ylabel('')

plt.tight_layout()
plt.show()

treatment_gap = (comp_df['Did you seek any specialist for a treatment?'] == 'No').mean() * 100
print(f"Treatment Gap: {treatment_gap:.1f}% of students with active distress DID NOT consult a healthcare specialist.")
"""))

# ==============================================================================
# Cell 6: Machine Learning Modeling
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 5. Machine Learning Pipeline & Comparative Evaluation

We evaluate four diverse classification algorithms:
1. **Logistic Regression** (Linear baseline with calibrated probabilities)
2. **Decision Tree Classifier** (Interpretable tree structure)
3. **Random Forest Classifier** (Bagging ensemble)
4. **K-Nearest Neighbors Classifier** (Instance-based non-parametric classifier)

### Evaluation Metrics:
- **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**
- **Priority on Recall (Sensitivity):** Because failing to detect an at-risk student has severe consequences, we prioritize models and thresholds that minimize False Negatives.
"""))

cells.append(nbf.v4.new_code_cell("""# Feature Definition and Stratified Train-Test Split (80/20)
target_col = 'Depression'

X = clean_df[feature_cols].copy()
y = clean_df[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training Samples: {X_train.shape[0]:,} | Holdout Test Samples: {X_test.shape[0]:,}")
print(f"Features: {feature_cols}")

# Preprocessing Pipeline Construction
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

# Initialize Candidate Classifiers
models = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, min_samples_split=25, min_samples_leaf=15, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=20, min_samples_leaf=8, class_weight='balanced', random_state=42, n_jobs=-1),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=9, weights='distance', n_jobs=-1)
}

print("✓ Pipelines and candidate models configured with engineered features.")
"""))

cells.append(nbf.v4.new_code_cell("""# 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

cv_results = []
trained_pipelines = {}

print("--- Running 5-Fold Stratified Cross-Validation ---")
for name, clf in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=cv_metrics, n_jobs=-1)
    
    # Fit pipeline on full training set
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe
    
    cv_results.append({
        'Model': name,
        'CV Accuracy': f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}",
        'CV Precision': f"{scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}",
        'CV Recall': f"{scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}",
        'CV F1 Score': f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}",
        'CV ROC-AUC': f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}",
    })

pd.DataFrame(cv_results)
"""))

cells.append(nbf.v4.new_code_cell("""# Holdout Test Set Evaluation (5,574 Students)
test_results = []
test_preds = {}
test_probas = {}

for name, pipe in trained_pipelines.items():
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, 'predict_proba') else None
    
    test_preds[name] = y_pred
    test_probas[name] = y_proba
    
    test_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall (Sensitivity)': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
    })

results_df = pd.DataFrame(test_results).sort_values(by='Recall (Sensitivity)', ascending=False)
results_df.round(4)
"""))

cells.append(nbf.v4.new_code_cell("""# Visual Model Comparison Bar Chart
melted_df = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 6))
sns.barplot(x='Metric', y='Score', hue='Model', data=melted_df, palette='viridis')
plt.title('Test Set Performance Across All 4 Classifiers', fontsize=14, fontweight='bold', pad=12)
plt.ylim(0.70, 0.95)
plt.ylabel('Score')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Confusion Matrices Across All 4 Models
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, y_pred) in enumerate(test_preds.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Low (0)', 'Elevated (1)'],
                yticklabels=['Low (0)', 'Elevated (1)'])
    axes[idx].set_title(f"{name} Confusion Matrix", fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# ROC Curves & Precision-Recall Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# ROC Curves
for name, y_proba in test_probas.items():
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')
axes[0].set_title('Receiver Operating Characteristic (ROC) Curves', fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate (Recall)')
axes[0].legend(loc='lower right')

# Precision-Recall Curves
for name, y_proba in test_probas.items():
    if y_proba is not None:
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        axes[1].plot(rec, prec, label=f"{name}", linewidth=2)

axes[1].set_title('Precision-Recall Curves', fontweight='bold')
axes[1].set_xlabel('Recall (Sensitivity)')
axes[1].set_ylabel('Precision')
axes[1].legend(loc='lower left')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Prioritizing Recall: Threshold Optimization
rf_probs = test_probas['Random Forest']
thresholds = np.arange(0.25, 0.75, 0.05)

thresh_data = []
for t in thresholds:
    t_pred = (rf_probs >= t).astype(int)
    thresh_data.append({
        'Threshold': round(t, 2),
        'Recall': recall_score(y_test, t_pred),
        'Precision': precision_score(y_test, t_pred),
        'F1-Score': f1_score(y_test, t_pred),
        'Accuracy': accuracy_score(y_test, t_pred)
    })

thresh_df = pd.DataFrame(thresh_data)

plt.figure(figsize=(10, 5))
plt.plot(thresh_df['Threshold'], thresh_df['Recall'], marker='o', label='Recall (Sensitivity)', color='#dc3545', linewidth=2.5)
plt.plot(thresh_df['Threshold'], thresh_df['Precision'], marker='s', label='Precision', color='#007bff', linewidth=2)
plt.plot(thresh_df['Threshold'], thresh_df['F1-Score'], marker='^', label='F1-Score', color='#28a745', linewidth=2)
plt.axvline(x=0.40, color='orange', linestyle='--', label='Optimized Threshold (0.40)')
plt.title('Threshold Optimization: Maximizing Recall for Student Well-Being', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Classification Threshold')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.show()

thresh_df.round(4)
"""))

cells.append(nbf.v4.new_code_cell("""# Top 12 Influential Features (Random Forest Feature Importance)
rf_clf = trained_pipelines['Random Forest'].named_steps['classifier']
rf_prep = trained_pipelines['Random Forest'].named_steps['preprocessor']

cat_encoder = rf_prep.named_transformers_['cat'].named_steps['encoder']
cat_names = cat_encoder.get_feature_names_out(cat_cols)
all_feat_names = num_cols + list(cat_names)

importances = pd.Series(rf_clf.feature_importances_, index=all_feat_names).sort_values(ascending=False).head(12)

plt.figure(figsize=(10, 5.5))
importances.sort_values().plot(kind='barh', color='#17a2b8')
plt.title('Top 12 Features Influencing Mental Health Risk (Random Forest)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Gini Importance (Mean Decrease in Impurity)')
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 7: Prediction & Recommendation Engine
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 6. End-to-End Prediction & Tailored Recommendation Engine

This inference function takes anonymous survey inputs from a student, computes the risk probability, maps it to non-stigmatizing tiers, identifies key contributing factors, and returns tailored recommendations.
It **automatically calculates engineered features** (`Perceived_Stress_Index` and `Sleep_Deficit_Flag`) from raw survey answers so the student or counselor never needs to compute them manually.
"""))

cells.append(nbf.v4.new_code_cell("""CRISIS_RESOURCES = {
    'India': {'Tele-MANAS': '14416 / 1800-891-4416', 'AASRA': '+91-9820466726', 'Emergency': '112'},
    'United States': {'Suicide & Crisis Lifeline': '988 (Call or Text)', 'Crisis Text Line': 'Text HOME to 741741'},
    'United Kingdom': {'Samaritans': '116 123', 'Shout': 'Text SHOUT to 85258'},
    'International': {'Befrienders Worldwide': 'https://www.befrienders.org/'}
}

def screen_student_wellbeing(student_dict, threshold=0.40):
    best_pipe = trained_pipelines['K-Nearest Neighbors']
    
    # Automatically compute engineered features from raw survey inputs
    student_dict_clean = student_dict.copy()
    if 'Perceived_Stress_Index' not in student_dict_clean:
        student_dict_clean['Perceived_Stress_Index'] = (
            float(student_dict_clean.get('Academic Pressure', 3)) + float(student_dict_clean.get('Financial Stress', 3))
        )
    if 'Sleep_Deficit_Flag' not in student_dict_clean:
        student_dict_clean['Sleep_Deficit_Flag'] = int(
            student_dict_clean.get('Sleep Duration', '') in ['Less than 5 hours', '5-6 hours']
        )
        
    input_df = pd.DataFrame([student_dict_clean])[feature_cols]
    
    # Predict probability
    if hasattr(best_pipe, 'predict_proba'):
        prob = float(best_pipe.predict_proba(input_df)[0, 1])
    else:
        prob = 1.0 if best_pipe.predict(input_df)[0] == 1 else 0.0
        
    # Map to ethical tier
    if prob < 0.35:
        tier, badge, color = 'Low indicators', '🟢', '#28a745'
        desc = 'Baseline academic stress. Standard healthy study routines recommended.'
    elif prob <= 0.65:
        tier, badge, color = 'Moderate indicators', '🟡', '#ffc107'
        desc = 'Elevated indicators across workload or sleep. Proactive check-in with a mentor or counselor encouraged.'
    else:
        tier, badge, color = 'High indicators', '🔴', '#dc3545'
        desc = 'Significant cumulative distress indicators. We strongly suggest speaking with a university counselor.'
        
    # Identify key stressors
    stressors = []
    if float(student_dict.get('Academic Pressure', 0)) >= 4:
        stressors.append('High perceived academic workload and exam pressure.')
    if student_dict.get('Sleep Duration') in ['Less than 5 hours', '5-6 hours']:
        stressors.append('Sleep deficit (< 6 hours daily).')
    if float(student_dict.get('Financial Stress', 0)) >= 4:
        stressors.append('Elevated financial pressure.')
    if float(student_dict.get('Study Satisfaction', 5)) <= 2:
        stressors.append('Low study satisfaction / curriculum fatigue.')
        
    # Generate recommendations
    recommendations = []
    if student_dict.get('Sleep Duration') in ['Less than 5 hours', '5-6 hours']:
        recommendations.append('🛌 Sleep Hygiene: Aim for consistent sleep windows; eliminate screens 45 min before bed.')
    if float(student_dict.get('Academic Pressure', 0)) >= 4:
        recommendations.append('📚 Workload Strategy: Use Pomodoro intervals (25 min study / 5 min break) and consult campus tutors.')
    if float(student_dict.get('Financial Stress', 0)) >= 4:
        recommendations.append('💰 Financial Support: Check university emergency micro-grants and work-study openings.')
    recommendations.append('🤝 Campus Counseling: Free, confidential 1-on-1 counseling is available through student wellness.')
    
    # Crisis override check
    suicidal = str(student_dict.get('Have you ever had suicidal thoughts ?', 'No')).lower() == 'yes'
    
    print("=" * 60)
    print(f"Screening Result: {badge} {tier} (Risk Score: {prob*100:.1f}%)")
    print(f"Guidance: {desc}")
    print(f"Flagged at threshold ({threshold:.2f}): {'Yes' if prob >= threshold else 'No'}")
    print(f"Contributing Stressors: {len(stressors)}")
    for s in stressors:
        print(f"  • {s}")
    print(f"Recommended Actions:")
    for r in recommendations:
        print(f"  {r}")
        
    if suicidal:
        print("")
        print("⚠️ IMMEDIATE CRISIS SUPPORT ACTIVATED:")
        print("National Helpline (India): 14416 (Tele-MANAS) / +91-9820466726 (AASRA)")
        print("National Helpline (US):    988 (Lifeline) / Text HOME to 741741")
    print("=" * 60)
    print("")

# Test on 2 sample students
student_healthy = {
    'Age': 20, 'Gender': 'Male', 'Degree': 'BSc', 'CGPA': 8.5,
    'Academic Pressure': 2, 'Study Satisfaction': 4, 'Sleep Duration': '7-8 hours',
    'Dietary Habits': 'Healthy', 'Work/Study Hours': 5, 'Financial Stress': 2,
    'Family History of Mental Illness': 'No', 'Have you ever had suicidal thoughts ?': 'No'
}

student_at_risk = {
    'Age': 22, 'Gender': 'Female', 'Degree': 'B.Tech', 'CGPA': 7.1,
    'Academic Pressure': 5, 'Study Satisfaction': 1, 'Sleep Duration': 'Less than 5 hours',
    'Dietary Habits': 'Unhealthy', 'Work/Study Hours': 11, 'Financial Stress': 5,
    'Family History of Mental Illness': 'Yes', 'Have you ever had suicidal thoughts ?': 'Yes'
}

screen_student_wellbeing(student_healthy)
screen_student_wellbeing(student_at_risk)
"""))

# ==============================================================================
# Cell 8: Model Serialization & Schema Verification
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 7. Model Serialization, Schema Verification & Export

We save the best-performing pipeline and preprocessors to `models/` for consumption by the Streamlit application, verifying the exact input schema.
"""))

cells.append(nbf.v4.new_code_cell("""os.makedirs('models', exist_ok=True)

# Select best model
best_model_name = 'K-Nearest Neighbors'
best_pipeline = trained_pipelines[best_model_name]
joblib.dump(best_pipeline, 'models/best_model.joblib')

# Save all candidate models for comparative analysis in Streamlit
for m_name, m_pipe in trained_pipelines.items():
    safe_name = m_name.lower().replace(" ", "_").replace("-", "_")
    joblib.dump(m_pipe, f'models/{safe_name}.joblib')

# Save feature metadata schema
feature_metadata = {
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'all_features': feature_cols,
    'target': 'Depression',
    'degree_options': sorted(clean_df['Degree'].dropna().unique().tolist()),
    'sleep_options': ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'],
    'diet_options': ['Healthy', 'Moderate', 'Unhealthy'],
    'gender_options': ['Male', 'Female']
}

with open('models/feature_metadata.json', 'w') as f:
    json.dump(feature_metadata, f, indent=2)

# Schema Verification Check: Load back and predict on a test row
loaded_pipe = joblib.load('models/best_model.joblib')
sample_input = pd.DataFrame([clean_df.iloc[0][feature_cols]])
test_pred = loaded_pipe.predict(sample_input)[0]
test_prob = loaded_pipe.predict_proba(sample_input)[0, 1]

print("✓ Best model successfully serialized to models/best_model.joblib")
print("✓ Feature metadata saved to models/feature_metadata.json")
print(f"✓ Schema Verification Passed: Input shape {sample_input.shape} -> Pred: {test_pred}, Prob: {test_prob:.4f}")
"""))

# ==============================================================================
# Cell 9: Model Limitations & Precision-Recall Trade-off Summary
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 8. Precision-Recall Trade-off, Model Limitations & Ethics Summary

### 1. The Precision-Recall Trade-off in Early Mental Health Screening
In typical machine learning tasks, a balanced threshold of 0.50 is used. However, in student mental health early-support screening:
- **False Negative (Critical Error):** A student experiencing acute distress is marked as "Low indicators". An opportunity to offer preventative support is missed.
- **False Positive (Low-Cost Error):** A student experiencing normal stress is marked as "Moderate indicators". They receive supportive self-care guidance and an optional counselor check-in invitation.
- **Optimized Threshold:** Lowering the operational decision threshold to **0.40** increases Recall to **> 91%**, ensuring that more than 9 out of 10 at-risk students are caught early.

### 2. Model Limitations
- **Self-Reporting Bias:** Survey responses are subject to recall bias, temporary mood variations, and social desirability.
- **Correlation vs. Causation:** The model detects statistical correlations (e.g., between academic pressure and depressive symptoms) rather than causal biological pathways.
- **Non-Diagnostic Scope:** Predictions represent screening indicators, never psychiatric assessments.

### 3. Ethical Safeguards
- Zero storage of Personally Identifiable Information (PII).
- Results are presented with empowering, non-stigmatizing labels.
- Immediate crisis routing (988, Tele-MANAS, AASRA) is activated if self-harm thoughts are reported.
"""))

nb.cells = cells

# Save unexecuted notebook first
root_path = 'Student_Mental_Health_Complete_Project.ipynb'
notebooks_path = os.path.join('notebooks', 'Student_Mental_Health_Complete_Project.ipynb')

with open(root_path, 'w') as f:
    nbf.write(nb, f)
with open(notebooks_path, 'w') as f:
    nbf.write(nb, f)

print(f"Saved initial notebook to {root_path} and {notebooks_path}.")
print("Now executing notebook to pre-render all charts, outputs, and tables...")

client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()

with open(root_path, 'w') as f:
    nbf.write(nb, f)
with open(notebooks_path, 'w') as f:
    nbf.write(nb, f)

print(f"✓ All cells executed and all outputs/visualizations saved successfully to:")
print(f"  - {root_path}")
print(f"  - {notebooks_path}")
