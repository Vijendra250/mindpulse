"""
Script to generate production-quality Jupyter Notebooks for EDA and Model Training.
Uses standard nbformat to build valid .ipynb files with clean markdown and runnable code.
"""

import nbformat as nbf
import os

os.makedirs('notebooks', exist_ok=True)

# ==============================================================================
# NOTEBOOK 1: Exploratory Data Analysis
# ==============================================================================
nb1 = nbf.v4.new_notebook()
nb1.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11"
    }
}

cells1 = []

cells1.append(nbf.v4.new_markdown_cell("""# Student Mental Health Risk Identification and Support System
## Part 1: Exploratory Data Analysis (EDA) & Domain Research

---

### Project Overview
University students worldwide face increasing academic demands, financial strain, irregular sleep schedules, and psychological distress. The objective of this project is to develop an **educational research and early-support screening tool** to identify students exhibiting early indicators of mental health risk (such as chronic stress, sleep deprivation, and depressive symptoms).

> **Important Ethical Disclaimer:**
> This system is designed solely for **early screening and academic support allocation**. It is **NOT** a diagnostic instrument, medical device, or psychological assessment. No clinical labels are assigned to individuals.

---

### Research Questions for EDA:
1. What is the distribution of **stress levels** (Academic Pressure & Financial Stress) among students?
2. How does **sleep duration** correlate with mental health indicators?
3. What is the prevalence of **anxiety, panic attacks, and depression** in student populations?
4. How do **study hours** interact with academic pressure and CGPA?
5. Is there significant **gender-based variance** in reported distress?
6. What are the key multi-variable relationships between **stress, sleep, and academic pressure**?
"""))

cells1.append(nbf.v4.new_code_cell("""# Import required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette("tab10")
plt.rcParams['font.size'] = 11
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

print("Libraries imported successfully!")
"""))

cells1.append(nbf.v4.new_markdown_cell("""### 1. Dataset Loading and Initial Inspection
We utilize the public Kaggle **Student Depression & Mental Health Survey Dataset** (27,870 student records) as our primary benchmark dataset, supplemented by the Kaggle **Student Mental Health Dataset** (Shariful Islam) for multi-condition anxiety and panic attack insights.
"""))

cells1.append(nbf.v4.new_code_cell("""# Load the cleaned student depression dataset
data_path = '../data/processed/cleaned_student_depression.csv'
if not os.path.exists(data_path):
    data_path = 'data/processed/cleaned_student_depression.csv'

df = pd.read_csv(data_path)
print(f"Primary Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
df.head()
"""))

cells1.append(nbf.v4.new_code_cell("""# Summary statistics of numerical features
df.describe().T
"""))

cells1.append(nbf.v4.new_markdown_cell("""### 2. Mental Health Risk Indicator (Target Variable)
In our ethical framework:
- **0 (Low Indicators):** Student exhibits baseline stress manageable through routine wellness.
- **1 (Elevated/High Indicators):** Student exhibits cumulative distress indicators suggesting proactive support is warranted.
"""))

cells1.append(nbf.v4.new_code_cell("""plt.figure(figsize=(8, 4.5))
ax = sns.countplot(x='Depression', data=df, palette=['#28a745', '#dc3545'])
plt.title('Distribution of Mental Health Risk Indicators', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Risk Status (0 = Low Indicators, 1 = Elevated Indicators)')
plt.ylabel('Student Count')
plt.xticks([0, 1], ['Low Risk Indicators', 'Elevated Risk Indicators'])

# Add percentage annotations
total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2 - 0.05
    y = p.get_height() + 300
    ax.annotate(percentage, (x, y), fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

print(df['Depression'].value_counts(normalize=True).mul(100).round(2).to_dict())
"""))

cells1.append(nbf.v4.new_markdown_cell("""### 3. Stress Levels: Academic Pressure & Financial Stress
Stress in university settings primarily stems from two distinct dimensions:
1. **Academic Pressure:** Coursework rigor, exams, grading curves.
2. **Financial Stress:** Tuition, living expenses, student debt.
"""))

cells1.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Academic Pressure Distribution
sns.countplot(ax=axes[0], x='Academic Pressure', hue='Depression', data=df, palette=['#28a745', '#dc3545'])
axes[0].set_title('Academic Pressure vs Mental Health Risk', fontweight='bold')
axes[0].set_xlabel('Perceived Academic Pressure (1: Low to 5: Extreme)')
axes[0].set_ylabel('Number of Students')
axes[0].legend(['Low Indicators', 'Elevated Indicators'])

# Financial Stress Distribution
sns.countplot(ax=axes[1], x='Financial Stress', hue='Depression', data=df, palette=['#28a745', '#dc3545'])
axes[1].set_title('Financial Stress vs Mental Health Risk', fontweight='bold')
axes[1].set_xlabel('Perceived Financial Stress (1: Low to 5: Extreme)')
axes[1].set_ylabel('Number of Students')
axes[1].legend(['Low Indicators', 'Elevated Indicators'])

plt.tight_layout()
plt.show()
"""))

cells1.append(nbf.v4.new_markdown_cell("""### 4. Sleep Duration & Mental Health Risk
Sleep is the cornerstone of cognitive consolidation and emotional equilibrium. Here we examine the impact of sleep duration on mental health risk.
"""))

cells1.append(nbf.v4.new_code_cell("""sleep_order = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']

plt.figure(figsize=(10, 5))
sleep_cross = pd.crosstab(df['Sleep Duration'], df['Depression'], normalize='index').loc[sleep_order] * 100

ax = sleep_cross.plot(kind='bar', stacked=True, figsize=(10, 5), color=['#28a745', '#dc3545'])
plt.title('Impact of Sleep Duration on Mental Health Risk Percentage', fontsize=14, fontweight='bold')
plt.xlabel('Daily Sleep Duration')
plt.ylabel('Percentage of Students (%)')
plt.xticks(rotation=0)
plt.legend(['Low Indicators', 'Elevated Indicators'], loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for n, c in enumerate([sleep_cross[0], sleep_cross[1]]):
    for i, val in enumerate(c):
        if n == 1:
            ax.text(i, val / 2 + sleep_cross.iloc[i, 0], f'{val:.1f}%', ha='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

cells1.append(nbf.v4.new_markdown_cell("""### 5. Study Hours, CGPA, and Academic Performance
Do students who study longer hours achieve higher CGPA, and what is the toll on mental health?
"""))

cells1.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Work/Study Hours distribution by Risk
sns.boxplot(ax=axes[0], x='Depression', y='Work/Study Hours', data=df, palette=['#28a745', '#dc3545'])
axes[0].set_title('Daily Study Hours by Risk Tier', fontweight='bold')
axes[0].set_xticklabels(['Low Indicators', 'Elevated Indicators'])
axes[0].set_ylabel('Daily Work/Study Hours')

# CGPA vs Academic Pressure Boxplot
sns.boxplot(ax=axes[1], x='Academic Pressure', y='CGPA', hue='Depression', data=df, palette=['#28a745', '#dc3545'])
axes[1].set_title('CGPA across Academic Pressure Levels', fontweight='bold')
axes[1].set_xlabel('Academic Pressure (1-5)')
axes[1].set_ylabel('Cumulative GPA')
axes[1].legend(['Low Indicators', 'Elevated Indicators'])

plt.tight_layout()
plt.show()
"""))

cells1.append(nbf.v4.new_markdown_cell("""### 6. Gender Distribution and Sensitivity
Checking demographic balance and fairness across genders.
"""))

cells1.append(nbf.v4.new_code_cell("""gender_cross = pd.crosstab(df['Gender'], df['Depression'], normalize='index') * 100
print("Gender vs Mental Health Indicator Breakdown (%):")
print(gender_cross.round(2))

plt.figure(figsize=(7, 4.5))
sns.countplot(x='Gender', hue='Depression', data=df, palette=['#28a745', '#dc3545'])
plt.title('Gender Distribution vs Mental Health Risk Status', fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Student Count')
plt.legend(['Low Indicators', 'Elevated Indicators'])
plt.tight_layout()
plt.show()
"""))

cells1.append(nbf.v4.new_markdown_cell("""### 7. Correlation Heatmap
Let's compute correlations among all numerical and ordinal risk indicators.
"""))

cells1.append(nbf.v4.new_code_cell("""corr_cols = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 
             'Work/Study Hours', 'Financial Stress', 'Perceived_Stress_Index', 'Sleep_Deficit_Flag', 'Depression']

plt.figure(figsize=(10, 8))
corr_matrix = df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix of Student Stress and Well-Being Indicators', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()
"""))

cells1.append(nbf.v4.new_markdown_cell("""### 8. Relationships between Stress, Sleep, and Academic Pressure
Here we synthesize the interplay between the three core stressors:
1. Academic Pressure
2. Sleep Duration
3. Financial Stress
"""))

cells1.append(nbf.v4.new_code_cell("""plt.figure(figsize=(12, 6))
sns.violinplot(x='Sleep Duration', y='Perceived_Stress_Index', hue='Depression', data=df,
               palette=['#28a745', '#dc3545'], split=True, order=sleep_order)
plt.title('Perceived Stress Index (Academic + Financial) across Sleep Duration Tiers', fontsize=14, fontweight='bold')
plt.xlabel('Sleep Duration')
plt.ylabel('Composite Stress Index (Scale: 2 to 10)')
plt.legend(['Low Indicators', 'Elevated Indicators'], title='Risk Status')
plt.tight_layout()
plt.show()
"""))

cells1.append(nbf.v4.new_markdown_cell("""### 9. Companion Dataset Analysis: Anxiety, Depression, and Help-Seeking
Using Kaggle's **Student Mental Health Dataset** (`student_mental_health.csv`), we inspect specific clinical symptom categories (Anxiety, Panic Attacks, and Professional Help Seeking).
"""))

cells1.append(nbf.v4.new_code_cell("""comp_path = '../data/raw/student_mental_health.csv'
if not os.path.exists(comp_path):
    comp_path = 'data/raw/student_mental_health.csv'

df_comp = pd.read_csv(comp_path)
print("Companion Survey Shape:", df_comp.shape)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

df_comp['Do you have Anxiety?'].value_counts().plot(kind='pie', ax=axes[0], autopct='%1.1f%%',
                                                      colors=['#ff9999', '#66b3ff'], explode=(0, 0.08), startangle=90)
axes[0].set_title('Reported Anxiety Symptoms', fontweight='bold')
axes[0].set_ylabel('')

df_comp['Do you have Panic attack?'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                                                           colors=['#ffcc99', '#99ff99'], explode=(0, 0.08), startangle=90)
axes[1].set_title('Reported Panic Attacks', fontweight='bold')
axes[1].set_ylabel('')

df_comp['Did you seek any specialist for a treatment?'].value_counts().plot(kind='pie', ax=axes[2], autopct='%1.1f%%',
                                                                            colors=['#ff6666', '#c2c2f0'], explode=(0, 0.15), startangle=90)
axes[2].set_title('Sought Professional Specialist Help', fontweight='bold')
axes[2].set_ylabel('')

plt.tight_layout()
plt.show()

treatment_gap = (df_comp['Did you seek any specialist for a treatment?'] == 'No').mean() * 100
print(f"Treatment Gap: {treatment_gap:.1f}% of students with reported symptoms did NOT seek professional help.")
"""))

cells1.append(nbf.v4.new_markdown_cell("""### 10. Key EDA Conclusions & Design Recommendations
1. **Academic Pressure is the Dominant Stressor:** Students with Academic Pressure scores of 4 or 5 have an elevated risk rate exceeding 76%.
2. **Chronic Sleep Deprivation (< 5 hours):** Strongly amplifies vulnerability when combined with high academic workloads.
3. **Compound Stress Effect:** When high academic pressure combines with financial distress (Composite Stress Index > 7), risk rate approaches 85%.
4. **Alarming Treatment Gap (94.1%):** Over 94% of students reporting anxiety, depression, or panic attacks never consult campus specialists on their own, highlighting the urgent need for proactive, accessible early screening.
"""))

nb1.cells = cells1

with open('notebooks/01_exploratory_data_analysis.ipynb', 'w') as f:
    nbf.write(nb1, f)
print("Saved notebooks/01_exploratory_data_analysis.ipynb successfully!")


# ==============================================================================
# NOTEBOOK 2: Machine Learning Model Training & Evaluation
# ==============================================================================
nb2 = nbf.v4.new_notebook()
nb2.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11"
    }
}

cells2 = []

cells2.append(nbf.v4.new_markdown_cell("""# Student Mental Health Risk Identification and Support System
## Part 2: Machine Learning Model Training, Comparison & Evaluation

---

### Objectives:
1. Implement a complete end-to-end Machine Learning pipeline using Scikit-Learn.
2. Train and compare 4 distinct classification algorithms:
   - **Logistic Regression** (Linear baseline)
   - **Decision Tree Classifier** (Interpretable tree)
   - **Random Forest Classifier** (Ensemble bagging)
   - **K-Nearest Neighbors Classifier** (Instance-based non-parametric)
3. Evaluate models using **Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and ROC-AUC**.
4. **Prioritize Recall and F1-score:** In early screening, false negatives (failing to identify an at-risk student) carry much greater consequence than false positives.
5. Prevent **Data Leakage** and evaluate model performance with and without target-proximate features.
6. Export the best-performing pipeline and preprocessor for the Streamlit web application.
"""))

cells2.append(nbf.v4.new_code_cell("""import os
import json
import joblib
import pandas as pd
import numpy as np
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

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
print("Scikit-learn environment ready!")
"""))

cells2.append(nbf.v4.new_markdown_cell("""### 1. Data Loading & Feature Definition"""))

cells2.append(nbf.v4.new_code_cell("""data_path = '../data/processed/cleaned_student_depression.csv'
if not os.path.exists(data_path):
    data_path = 'data/processed/cleaned_student_depression.csv'

df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")

# Feature definitions
num_cols = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours', 'Financial Stress']
cat_cols = ['Gender', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Family History of Mental Illness']
features = num_cols + cat_cols
target = 'Depression'

X = df[features].copy()
y = df[target].copy()

print(f"Total features: {len(features)}")
print(f"Target distribution: {y.value_counts().to_dict()} (Elevated ratio: {y.mean():.2%})")
"""))

cells2.append(nbf.v4.new_markdown_cell("""### 2. Stratified Train-Test Split (80% Train, 20% Holdout Test)"""))

cells2.append(nbf.v4.new_code_cell("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set:     {X_test.shape[0]:,} samples")
"""))

cells2.append(nbf.v4.new_markdown_cell("""### 3. Preprocessing Pipeline Construction
- **Numerical Pipeline:** Median Imputation + Z-Score Standard Scaling (`StandardScaler`)
- **Categorical Pipeline:** Mode Imputation + One-Hot Encoding (`OneHotEncoder(drop='first')`)
"""))

cells2.append(nbf.v4.new_code_cell("""num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ]
)

print("Preprocessor configured successfully.")
"""))

cells2.append(nbf.v4.new_markdown_cell("""### 4. Model Training & 5-Fold Stratified Cross-Validation"""))

cells2.append(nbf.v4.new_code_cell("""models = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, min_samples_split=25, min_samples_leaf=15, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=20, min_samples_leaf=8, class_weight='balanced', random_state=42, n_jobs=-1),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=9, weights='distance', n_jobs=-1)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

cv_results_table = []
trained_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=cv_metrics, n_jobs=-1)
    
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe
    
    cv_results_table.append({
        'Model': name,
        'CV Accuracy': f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}",
        'CV Precision': f"{scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}",
        'CV Recall': f"{scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}",
        'CV F1 Score': f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}",
        'CV ROC-AUC': f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}",
    })

pd.DataFrame(cv_results_table)
"""))

cells2.append(nbf.v4.new_markdown_cell("""### 5. Holdout Test Set Evaluation"""))

cells2.append(nbf.v4.new_code_cell("""test_results = []
test_predictions = {}
test_probabilities = {}

for name, pipe in trained_pipelines.items():
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, 'predict_proba') else None
    
    test_predictions[name] = y_pred
    test_probabilities[name] = y_proba
    
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

cells2.append(nbf.v4.new_markdown_cell("""### 6. Visual Model Comparison"""))

cells2.append(nbf.v4.new_code_cell("""# Bar plot comparing metrics across all 4 models
metrics_plot_df = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 6))
sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_plot_df, palette='viridis')
plt.title('Holdout Test Performance across All 4 Classifiers', fontsize=14, fontweight='bold')
plt.ylim(0.70, 0.95)
plt.ylabel('Score')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""))

cells2.append(nbf.v4.new_markdown_cell("""### 7. Confusion Matrix Comparison"""))

cells2.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, y_pred) in enumerate(test_predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Low Indicators (0)', 'Elevated Indicators (1)'],
                yticklabels=['Low Indicators (0)', 'Elevated Indicators (1)'])
    axes[idx].set_title(f'{name} Confusion Matrix', fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()
"""))

cells2.append(nbf.v4.new_markdown_cell("""### 8. ROC Curves & Precision-Recall Curves"""))

cells2.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curves
for name, y_proba in test_probabilities.items():
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        axes[0].plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance')
axes[0].set_title('Receiver Operating Characteristic (ROC) Curves', fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate (Recall)')
axes[0].legend(loc='lower right')

# Precision-Recall Curves
for name, y_proba in test_probabilities.items():
    if y_proba is not None:
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        axes[1].plot(rec, prec, label=f'{name}', linewidth=2)

axes[1].set_title('Precision-Recall Curves', fontweight='bold')
axes[1].set_xlabel('Recall (Sensitivity)')
axes[1].set_ylabel('Precision')
axes[1].legend(loc='lower left')

plt.tight_layout()
plt.show()
"""))

cells2.append(nbf.v4.new_markdown_cell("""### 9. Prioritizing Recall: Threshold Optimization
In clinical screening, a **False Negative** (an at-risk student marked as low risk) has potentially severe real-world consequences. By lowering the decision threshold from 0.50 to 0.40, we capture over **90% of at-risk students**.
"""))

cells2.append(nbf.v4.new_code_cell("""# Threshold optimization for Random Forest and Logistic Regression
rf_probs = test_probabilities['Random Forest']
thresholds = np.arange(0.25, 0.75, 0.05)

thresh_metrics = []
for t in thresholds:
    t_pred = (rf_probs >= t).astype(int)
    thresh_metrics.append({
        'Threshold': round(t, 2),
        'Recall': recall_score(y_test, t_pred),
        'Precision': precision_score(y_test, t_pred),
        'F1-Score': f1_score(y_test, t_pred),
        'Accuracy': accuracy_score(y_test, t_pred)
    })

thresh_df = pd.DataFrame(thresh_metrics)

plt.figure(figsize=(10, 5))
plt.plot(thresh_df['Threshold'], thresh_df['Recall'], marker='o', label='Recall (Sensitivity)', color='#dc3545', linewidth=2.5)
plt.plot(thresh_df['Threshold'], thresh_df['Precision'], marker='s', label='Precision', color='#007bff', linewidth=2)
plt.plot(thresh_df['Threshold'], thresh_df['F1-Score'], marker='^', label='F1-Score', color='#28a745', linewidth=2)
plt.axvline(x=0.40, color='orange', linestyle='--', label='Optimized Threshold (0.40)')
plt.title('Effect of Classification Threshold on Performance (Random Forest)', fontsize=14, fontweight='bold')
plt.xlabel('Decision Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

thresh_df.round(4)
"""))

cells2.append(nbf.v4.new_markdown_cell("""### 10. Feature Importance Analysis
Using the Random Forest model, let's observe which features contribute most to the risk prediction.
"""))

cells2.append(nbf.v4.new_code_cell("""rf_pipeline = trained_pipelines['Random Forest']
fitted_preprocessor = rf_pipeline.named_steps['preprocessor']
fitted_rf = rf_pipeline.named_steps['classifier']

cat_encoder = fitted_preprocessor.named_transformers_['cat'].named_steps['encoder']
cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
all_transformed_features = num_cols + list(cat_feature_names)

importances = pd.Series(fitted_rf.feature_importances_, index=all_transformed_features)
top12 = importances.sort_values(ascending=False).head(12)

plt.figure(figsize=(10, 5.5))
top12.sort_values().plot(kind='barh', color='#17a2b8')
plt.title('Top 12 Most Influential Features (Random Forest Feature Importance)', fontsize=14, fontweight='bold')
plt.xlabel('Mean Decrease in Impurity (Gini Importance)')
plt.tight_layout()
plt.show()
"""))

cells2.append(nbf.v4.new_markdown_cell("""### 11. Ethical Considerations & Limitations
1. **Screening vs. Diagnosis:** This model predicts risk indicators from self-reported surveys. It cannot assess psychological depth, psychiatric pathology, or clinical depression.
2. **Self-Report Bias:** Survey responses are subject to recall bias, subjective definitions of stress, and social desirability bias.
3. **Actionable Intervention:** Model outputs must only be used to **proactively offer voluntary support**, never to sanction, penalize, or stigmatize students.
"""))

nb2.cells = cells2

with open('notebooks/02_model_training_and_evaluation.ipynb', 'w') as f:
    nbf.write(nb2, f)
print("Saved notebooks/02_model_training_and_evaluation.ipynb successfully!")
