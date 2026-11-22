# MindPulse: Student Mental Health Risk Identification and Support System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ethics: Non-Diagnostic](https://img.shields.io/badge/Ethics-Non--Diagnostic%20Screening-green.svg)](#privacy--ethics-framework)

An end-to-end Machine Learning and Educational Research project designed to detect early indicators of academic stress, chronic sleep debt, and psychological distress among university students using anonymized survey responses.

---

## ⚠️ Important Ethical & Clinical Disclaimer
> **This system is strictly an educational research and early-support screening tool. It does NOT provide clinical or medical diagnoses, psychiatric evaluations, or treatment prescriptions.**
> 
> Outputs are strictly categorized into non-stigmatizing tiers: **"Low indicators"**, **"Moderate indicators"**, or **"High indicators"**. Predictions are designed to empower students with personalized self-care recommendations and facilitate voluntary connection with qualified university counselors and licensed mental health professionals.

---

## 📌 Table of Contents
1. [Project Motivation & Objectives](#project-motivation--objectives)
2. [Datasets & Data Provenance](#datasets--data-provenance)
3. [Data Preprocessing & Leakage Prevention](#data-preprocessing--leakage-prevention)
4. [Exploratory Data Analysis (EDA) Highlights](#exploratory-data-analysis-eda-highlights)
5. [Machine Learning Pipeline & Comparative Analysis](#machine-learning-pipeline--comparative-analysis)
6. [Why Recall & F1-Score Were Prioritized](#why-recall--f1-score-were-prioritized)
7. [Streamlit Web Application](#streamlit-web-application)
8. [Privacy & Ethics Framework](#privacy--ethics-framework)
9. [Project Directory Structure](#project-directory-structure)
10. [Setup & Quickstart Guide](#setup--quickstart-guide)
11. [Portfolio & Technical Interview Highlights](#portfolio--technical-interview-highlights)

---

## 1. Project Motivation & Objectives
University students navigate intense academic expectations, sudden independence, financial strain, and irregular sleep patterns. Peer-reviewed studies consistently show that over **90% of students experiencing psychological distress never proactively visit campus counseling centers** due to perceived stigma, lack of awareness, or academic time constraints.

### Core Objectives:
- **Early-Warning Screening:** Identify students experiencing cumulative stress factors (sleep deprivation, academic overload, financial strain) before crises develop.
- **Explainable Predictions:** Attribute risk scores to concrete, actionable factors (e.g., `< 5 hours sleep`, `Pomodoro pacing`, `financial aid resources`).
- **Privacy by Design:** Collect zero Personally Identifiable Information (PII) and store zero persistent records.
- **Safety Protocol:** Provide immediate 24/7 crisis hotline routing if extreme distress or self-harm thoughts are reported.

---

## 2. Datasets & Data Provenance
This project utilizes authentic, publicly accessible survey datasets from Kaggle. No synthetic or fabricated data is used.

### Primary Dataset: Student Depression & Mental Health Survey
- **Source:** [Kaggle Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) (also benchmarked in Kaggle Playground Series s4e11).
- **Target Population:** University students across undergraduate, postgraduate, and professional degrees.
- **Cleaned Sample Size:** 27,870 student respondents.
- **Key Features:** `Age`, `Gender`, `Degree`, `CGPA`, `Academic Pressure` (1–5), `Study Satisfaction` (1–5), `Sleep Duration` (`< 5 hrs`, `5-6 hrs`, `7-8 hrs`, `> 8 hrs`), `Dietary Habits` (`Healthy`, `Moderate`, `Unhealthy`), `Work/Study Hours`, `Financial Stress` (1–5), `Family History of Mental Illness`.
- **Target:** `Depression` indicator (`0 = Low indicators`, `1 = Elevated indicators`).

### Companion Dataset: Student Mental Health Survey (Anxiety & Help-Seeking)
- **Source:** [Kaggle Student Mental Health by Shariful Islam](https://www.kaggle.com/datasets/shariful07/student-mental-health).
- **Key Features:** Clinical indicators including `Do you have Anxiety?`, `Do you have Panic attack?`, and `Did you seek any specialist for a treatment?`.
- **Key Finding:** Revealed an alarming **94.1% treatment-seeking gap** among students reporting acute distress.

---

## 3. Data Preprocessing & Leakage Prevention

### Preprocessing Pipeline (`src/data_preprocessing.py`)
1. **Population Filtering:** Filtered raw dataset to verified students (`Profession == 'Student'`), dropping 31 non-student occupational records.
2. **Feature Pruning & Leakage Prevention:**
   - Dropped arbitrary identifier `id` (prevents shortcut learning).
   - Dropped `City` (high cardinality noise preventing geographic bias).
   - Dropped `Work Pressure` and `Job Satisfaction` (strictly zero for full-time students).
3. **Handling Non-Standard Missing Values:**
   - Identified non-standard placeholder `'?'` in `Financial Stress` (3 occurrences). Coerced via `pd.to_numeric(..., errors='coerce')` and imputed with the median.
4. **Categorical Standardization:**
   - Standardized string casing, stripped quotations, and grouped sparse categories into `'Others'`.
5. **Feature Engineering:**
   - `Perceived_Stress_Index` = `Academic Pressure` + `Financial Stress` (scale 2 to 10).
   - `Sleep_Deficit_Flag` = Binary indicator for `< 6 hours` of daily sleep.
6. **Outlier Detection:**
   - Evaluated all numerical features using the **Interquartile Range (IQR) method** ($[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$).
   - Outliers represented $< 0.05\%$ of data (mature students up to age 59, freshman CGPAs of 0.0), retained to preserve model generalizability.

### Decoupling Suicidal Ideation from ML Training (Ethical ML Architecture)
- While `Have you ever had suicidal thoughts ?` correlates with distress, treating it as an ordinary ML feature creates **target-proxy leakage** (the model ignores lifestyle cues and simply learns whether the student reported suicidal thoughts).
- **Our Ethical Solution:** The ML model is trained strictly on **lifestyle, academic, and biological stressors**. If a student indicates suicidal thoughts on the survey form, the system triggers an immediate **Crisis Support Override Protocol** (providing 24/7 hotlines) regardless of model probability.

---

## 4. Exploratory Data Analysis (EDA) Highlights

The full statistical exploration is documented in [`notebooks/01_exploratory_data_analysis.ipynb`](notebooks/01_exploratory_data_analysis.ipynb).

| Analysis Dimension | Key Empirical Finding | Institutional Implication |
| :--- | :--- | :--- |
| **Academic Pressure** | At pressure levels 4 & 5, elevated risk rates surge to **76.1%** and **86.1%**. | Course pacing and exam schedules require structured wellness breaks. |
| **Sleep Duration** | Students sleeping `< 5 hours` exhibit an elevated risk rate of **64.5%**. | Chronic sleep deprivation directly impairs stress coping mechanisms. |
| **Compound Stress Effect** | When `Academic Pressure` $\ge 4$ and `Financial Stress` $\ge 4$, risk rate exceeds **84.8%**. | Financial strain acts as a continuous background multiplier of academic distress. |
| **Study Hours Sweet Spot** | Moderate study hours (4–7 hrs/day) associate with the lowest risk. Extremes ($> 10$ hrs/day) spike burnout indicators. | Encouraging balanced study habits over unbroken cramming protects well-being. |
| **Treatment-Seeking Gap** | **94.1%** of students with active anxiety or depression symptoms in the companion survey never consulted a specialist. | Universal, non-stigmatizing early screening is urgently needed. |

---

## 5. Machine Learning Pipeline & Comparative Analysis

We trained, tuned, and evaluated four distinct classification algorithms using **Stratified 5-Fold Cross-Validation** and a holdout test set (20% split, 5,574 unseen students).

```
                      ┌──────────────────────────────────────────────┐
                      │ Clean Student Survey Data (27,870 records)   │
                      └──────────────────────┬───────────────────────┘
                                             │
                                ┌────────────┴────────────┐
                                ▼                         ▼
                     Training Set (80%)           Test Set (20%)
                     (22,296 students)            (5,574 students)
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
       Numerical Pipeline            Categorical Pipeline
       - Median Imputer              - Most Frequent Imputer
       - StandardScaler              - OneHotEncoder(drop='first')
                 └──────────────┬──────────────┘
                                ▼
         5-Fold Stratified Cross-Validation & Grid Tuning
         ┌─────────────────────────────────────────────┐
         │ 1. Logistic Regression (Linear Baseline)    │
         │ 2. Decision Tree Classifier (Interpretable) │
         │ 3. Random Forest (Bagging Ensemble)         │
         │ 4. K-Nearest Neighbors (Instance-Based)     │
         └─────────────────────────────────────────────┘
```

### Benchmark Results on Holdout Test Set (5,574 Students)

| Model | Accuracy | Precision | **Recall (Sensitivity)** | **F1-Score** | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Nearest Neighbors (k=9)** | 77.22% | 78.77% | **83.60%** | **0.8111** | 0.8312 |
| **Random Forest (150 Trees)** | 78.67% | 82.92% | **80.04%** | **0.8145** | 0.8649 |
| **Logistic Regression** | 78.81% | 83.55% | **79.43%** | **0.8144** | 0.8671 |
| **Decision Tree (depth=6)** | 77.00% | 81.94% | **77.87%** | **0.7985** | 0.8434 |

---

## 6. Why Recall & F1-Score Were Prioritized

In typical binary classification, models are evaluated purely on Accuracy. In mental health screening, **classification errors are asymmetric**:
- **False Positive:** A student experiencing manageable stress is flagged with elevated indicators. **Outcome:** They receive supportive self-care tips and an optional invitation to talk with a counselor (minimal harm).
- **False Negative:** A severely distressed student is classified as "Low indicators". **Outcome:** A genuine opportunity for early intervention is missed, potentially allowing distress to escalate unaddressed (critical harm).

### Sensitivity Threshold Optimization
By shifting the classification decision threshold from the default $0.50$ to $0.40$:
- **Recall increases from 80.0% to > 91.2%**.
- More than **9 out of 10 at-risk students are successfully identified**.
- The Streamlit application includes an interactive **Threshold Slider** allowing educators and counselors to adjust this operational sensitivity dynamically.

---

## 7. Streamlit Web Application

The interactive web dashboard (`app.py`) provides a complete, modern user experience:

1. **Overview & Ethics:** Clear project statement, architectural overview, technology breakdown, and clinical disclaimer.
2. **Well-being Self-Screening Form:**
   - Anonymous input: Age, Gender, Degree, CGPA, Academic Pressure, Study Satisfaction, Sleep Duration, Dietary Habits, Study Hours, Financial Stress, and Family History.
   - Interactive gauge chart displaying risk probability.
   - Non-stigmatizing badges: `🟢 Low indicators`, `🟡 Moderate indicators`, `🔴 High indicators`.
   - **Key Contributing Stressors:** Itemized explanation of which factors drove the score.
   - **Personalized Recommendations:** Actionable guidance for sleep hygiene, Pomodoro academic pacing, financial counseling, and departmental advisors.
   - **Immediate Crisis Alert:** Instant helpline display if self-harm indicators are reported.
3. **Dataset Analytics & EDA:** Interactive Plotly charts for stress vs. risk, sleep duration distributions, study hours boxplots, and companion clinical surveys.
4. **Model Comparison & Benchmarks:** Comprehensive benchmark tables, confusion matrices, ROC curves, Random Forest feature importance, and interactive threshold sensitivity simulator.
5. **Privacy & Support Resources:** Full architectural privacy guarantee and international 24/7 crisis helplines for India, United States, United Kingdom, and global directories.

---

## 8. Privacy & Ethics Framework

| Principle | Implementation in MindPulse |
| :--- | :--- |
| **Zero PII Collection** | No names, phone numbers, email addresses, student IDs, or IP addresses are requested or logged. |
| **RAM-Only Processing** | Survey inputs are processed purely in volatile server memory during the user session and discarded immediately. No database persistence exists. |
| **Non-Diagnostic Vocabulary** | Never uses labels like "depressed", "ill", or "abnormal". Results are strictly presented as *risk indicator levels* for voluntary exploration. |
| **Crisis Intervention Safeguard** | Instant 24/7 helpline routing (988, AASRA, Tele-MANAS, Crisis Text Line) if critical distress is reported. |

---

## 9. Project Directory Structure

```
student_mental_health_project/
│
├── data/
│   ├── raw/
│   │   ├── student_depression_dataset.csv     # Kaggle primary survey (27,901 rows)
│   │   └── student_mental_health.csv          # Kaggle companion clinical survey (101 rows)
│   └── processed/
│       └── cleaned_student_depression.csv     # Cleaned student population (27,870 rows)
│
├── Student_Mental_Health_Complete_Project.ipynb # All-in-one executed end-to-end notebook
├── notebooks/
│   ├── Student_Mental_Health_Complete_Project.ipynb # All-in-one executed notebook (mirror)
│   ├── 01_exploratory_data_analysis.ipynb     # Statistical EDA & visualizations
│   ├── 02_model_training_and_evaluation.ipynb # Full ML training, CV & comparisons
│   └── build_single_notebook.py               # Notebook build and execution pipeline
│
├── src/
│   ├── __init__.py                            # Package init
│   ├── data_preprocessing.py                  # Data cleaning & outlier detection pipeline
│   ├── train.py                               # 5-fold CV training of all 4 models
│   ├── evaluate.py                            # Comprehensive evaluation & metric reports
│   ├── predict.py                             # Inference class with explainable factors
│   └── utils.py                               # Risk tiers, recommendation engine, crisis numbers
│
├── models/
│   ├── best_model.joblib                      # Serialized best ML pipeline
│   ├── logistic_regression.joblib             # Serialized Logistic Regression
│   ├── decision_tree.joblib                   # Serialized Decision Tree
│   ├── random_forest.joblib                   # Serialized Random Forest
│   ├── k_nearest_neighbors.joblib             # Serialized KNN
│   ├── feature_metadata.json                  # Schema and categorical mappings
│   └── model_metrics.json                     # Test & cross-validation metrics
│
├── app.py                                     # Interactive Streamlit Web Application
├── requirements.txt                           # Production dependency specification
└── README.md                                  # Complete project documentation
```

---

## 10. Setup & Quickstart Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12 (compatible through 3.14)
- Git

### 1. Clone & Navigate to Repository
```bash
git clone <repository_url>
cd "Student Mental Health"
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
# On macOS / Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Machine Learning Pipeline
To execute data preprocessing, train all 4 classifiers, perform 5-fold cross-validation, and export models:
```bash
python3 src/train.py
```

### 5. Launch the Streamlit Web Dashboard
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

### 6. Run the Complete Jupyter Notebook
To interactively view the full end-to-end pipeline with all pre-rendered charts and model evaluations:
```bash
jupyter notebook Student_Mental_Health_Complete_Project.ipynb
```
Or open in VS Code:
```bash
code .
```
And select the pre-configured kernel: **`Python (.venv) (Student Mental Health)`**.

---

## 11. Portfolio & Technical Interview Highlights

When presenting this project to recruiters and engineering managers, highlight these key design choices:

1. **Why Prioritize Recall over Accuracy?**
   - *Explanation:* In medical and student well-being screening, false negatives carry real human consequences. We intentionally tuned decision thresholds to maximize recall ($> 90\%$) while maintaining high precision ($\approx 80\%$).
2. **How Did You Avoid Data Leakage?**
   - *Explanation:* We removed the `id` column, non-student occupational fields, and explicitly decoupled self-reported suicidal ideation from ML training. This prevented the model from taking shortcuts and forced it to learn meaningful signals from sleep habits, academic pressure, and study hours.
3. **How Did You Handle Non-Standard Missing Values?**
   - *Explanation:* Standard `.isnull()` calls failed to detect string-encoded question marks (`'?'`) in `Financial Stress`. We implemented robust type coercion (`pd.to_numeric(..., errors='coerce')`) followed by median imputation.
4. **Why Choose a Modular `src/` Architecture?**
   - *Explanation:* Production ML systems separate data preprocessing, training, evaluation, and inference into distinct, testable modules rather than keeping everything in monolithic notebooks.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details. Built for educational research and early student support screening.
