<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&duration=3000&pause=1000&color=C0392B&center=true&width=700&lines=Heart+Disease+Risk+Prediction;+%7C+Ensemble+Learning+%7C+Explainable+AI" alt="Typing SVG" />

<br/>

# 🫀 Heart Disease Risk Prediction
### *Multi-Center Dataset Fusion · Stacking Ensemble · SHAP + LIME Explainability*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/azeemalley)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-Tuned-2EC4B6?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-FF4B4B?style=for-the-badge)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)](LICENSE)

<br/>

> **"Every year, 17.9 million people die from heart disease — most of them preventably."**
>
> This project asks: *Can a machine learning model predict cardiac risk from a routine clinical checkup alone?*
>
> The answer is **yes — with 93.2% accuracy and 0.965 AUC.**

<br/>

---

</div>

## 📌 Table of Contents
- [🎯 Project Overview](#-project-overview)
- [💡 Why This Project Stands Out](#-why-this-project-stands-out)
- [📊 Dataset](#-dataset)
- [🏗️ Project Architecture](#️-project-architecture)
- [📂 Repository Structure](#-repository-structure)
- [🔬 Methodology — All 6 Phases](#-methodology--all-6-phases)
- [🧠 Feature Engineering](#-feature-engineering)
- [🏆 Results](#-results)
- [📈 Comparison with Related Work](#-comparison-with-related-work)
- [🔍 Explainability — SHAP & LIME](#-explainability--shap--lime)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 How to Run](#-how-to-run)
- [📄 Research Paper](#-research-paper)
- [👤 Author](#-author)

---

## 🎯 Project Overview

Cardiovascular disease is the **#1 cause of death globally**. In Pakistan alone, it accounts for **28% of all deaths**. Current diagnosis requires expensive procedures like coronary angiography (PKR 50,000–100,000) — out of reach for most patients.

**This project builds a full ML pipeline** that predicts heart disease from routine, non-invasive clinical tests available at any basic health facility — making early cardiac risk screening accessible to everyone.

```
Raw Dataset (920 patients, 4 Clinical Centers)
            ↓
    EDA + Visualization (8+ professional plots)
            ↓
  Preprocessing (imputation, scaling, encoding)
            ↓
  Feature Engineering (8 Clinical Features + K-Means)
            ↓
  Stacking Ensemble (RF + XGBoost + LightGBM + SVM)
            ↓
      Optuna Hyperparameter Optimization
            ↓
    SHAP + LIME Dual Explainability Layer
            ↓
    93.2% Accuracy  |  0.965 ROC-AUC ✅
```

---

## 💡 Why This Project Stands Out

Unlike typical ML projects that run a few models on the Cleveland 303-row dataset, this work implements **6 methodological innovations** not found together in any of the 5 compared published papers:

| Innovation | What Most Papers Do | What We Do |
|---|---|---|
| 📦 **Dataset** | Cleveland only (303 rows) | All 4 UCI centers combined (**920 rows**) |
| 🔧 **Features** | Raw features only | **8 clinically-informed** engineered features |
| 🏗️ **Modeling** | Compare models independently | **Stacking ensemble** (RF + XGB + LGBM + SVM) |
| ⚙️ **Tuning** | Grid search or none | **Optuna Bayesian optimization** (50 trials) |
| ⚖️ **Imbalance** | SMOTE or nothing | **SMOTE + Tomek Links** (cleaner boundaries) |
| 🔍 **Explainability** | None or basic bar charts | **Full SHAP (global + patient-level) + LIME** |

---

## 📊 Dataset

| Field | Details |
|---|---|
| **Name** | UCI Heart Disease Dataset (Combined 4-Center) |
| **Kaggle Link** | [redwankarimsony/heart-disease-data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) |
| **Total Rows** | **920 patients** |
| **Features** | 13 clinical input features + 1 target |
| **Task** | Binary Classification (Heart Disease: Yes / No) |
| **Centers** | Cleveland · Hungary · Switzerland · VA Long Beach |

### 🏥 Clinical Features

| Feature | Description | Type |
|---|---|---|
| `age` | Patient age in years | Numeric |
| `sex` | 1 = Male, 0 = Female | Binary |
| `cp` | Chest pain type (0 = Asymptomatic → 3 = Typical Angina) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numeric |
| `chol` | Serum cholesterol (mg/dl) | Numeric |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary |
| `restecg` | Resting ECG results (0–2) | Categorical |
| `thalach` | Maximum heart rate achieved during stress test | Numeric |
| `exang` | Exercise-induced angina (1 = Yes) | Binary |
| `oldpeak` | ST depression during exercise relative to rest | Numeric |
| `slope` | ST segment slope (0 = Upsloping, 1 = Flat, 2 = Down) | Categorical |
| `ca` | Number of blocked coronary vessels colored by fluoroscopy (0–3) | Numeric |
| `thal` | Thalassemia perfusion type (0 = Normal, 2 = Reversible Defect) | Categorical |
| `target` | **Heart disease present: 1 = Yes, 0 = No** | **Target** |

---

## 🏗️ Project Architecture

```
┌───────────────────────────────────────────────────────────────┐
│              PHASE 1: Problem Identification                  │
│  Dataset Selection · Research Question · Novelty Proposal     │
└──────────────────────────┬────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────┐
│              PHASE 2: Visualization & Preprocessing           │
│  EDA · Statistical Analysis · 8+ Plots · Outlier Treatment    │
│  RobustScaler · Encoding · Save Preprocessed CSV              │
└──────────────────────────┬────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────┐
│              PHASE 3: Feature Engineering                     │
│  5-Method Importance Consensus · SHAP Analysis                │
│  8 Clinical Features · K-Means Cluster · Validate Impact      │
└──────────────────────────┬────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────┐
│              PHASE 4: Modeling & Results                      │
│  SMOTE+Tomek · 5 Base Models · 10-Fold Stratified CV          │
│  Optuna HPO (50 trials) · Stacking Ensemble · Full Eval       │
└──────────────────────────┬────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────┐
│              PHASE 5: Research Comparison                     │
│  5 Papers Reviewed (2022–2024) · Gap Analysis · Radar Chart   │
│  LIME Explanations · Performance Comparison Charts            │
└──────────────────────────┬────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────┐
│              PHASE 6: Research Paper                          │
│  IEEE-Style · 7 Sections · 12 References · Clinical Insights  │
└───────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
heart-disease-ml/
│
├── 📓 notebooks/
│   ├── Phase1_Problem_Identification.ipynb
│   ├── Phase2_Visualization_Preprocessing.ipynb
│   ├── Phase3_Feature_Engineering.ipynb
│   ├── Phase4_Methodology_Results.ipynb
│   └── Phase5_Research_Comparison.ipynb
│
├── 📊 data/
│   ├── heart_disease_uci.csv                ← Raw dataset
│   ├── heart_disease_preprocessed.csv       ← After Phase 2
│   └── heart_disease_engineered.csv         ← After Phase 3 (model-ready)
│
├── 🖼️ images/
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_importance_5methods.png
│   ├── shap_beeswarm.png
│   ├── shap_waterfall_patient.png
│   ├── lime_explanation.png
│   ├── roc_curves.png
│   └── research_comparison_radar.png
│
├── 📄 paper/
│   └── Heart_Disease_Research_Paper.docx    ← IEEE-style research paper
│
└── 📋 README.md
```

---

## 🔬 Methodology — All 6 Phases

<details>
<summary><b>📘 Phase 1 — Problem Identification</b></summary>
<br/>

- Selected the **combined 4-center UCI dataset** (920 records) — 3× more data than most published papers
- Formulated a precise research question targeting a real clinical gap
- Proposed 5 novel contributions before writing a single line of code
- Documented full 6-phase pipeline with tools, libraries, and expected performance targets

</details>

<details>
<summary><b>📗 Phase 2 — Visualization & Preprocessing</b></summary>
<br/>

**Statistical Analysis:**
- `.describe()`, `.info()`, skewness and kurtosis for all features
- Missing value heatmap — identified impossible zero values in `chol` and `trestbps`
- Duplicate detection and class distribution analysis

**8+ Professional Visualizations:**
- Target distribution (pie + bar)
- Histograms with KDE overlay split by disease status
- Box plots for outlier visualization per feature
- Violin plots for distribution shape comparison
- Correlation heatmap (full matrix + feature-vs-target bar)
- Scatter: Age vs MaxHR colored by target with trend lines
- Categorical count plots (% disease by category for 8 features)
- Pairplot of top 4 predictive features

**Preprocessing Steps (each with written justification):**
1. Duplicate removal
2. Impossible zero imputation (median replacement for `chol`, `trestbps`)
3. IQR winsorization at 3× threshold — conservative to preserve medical extremes
4. One-hot encoding for ChestPainType (no ordinal assumption)
5. RobustScaler for continuous features (median+IQR — resistant to remaining outliers)
6. Preprocessed dataset saved as CSV for Phase 3

</details>

<details>
<summary><b>📙 Phase 3 — Feature Engineering</b></summary>
<br/>

**5-Method Feature Importance Consensus (reduces single-algorithm bias):**
- Random Forest (impurity-based)
- XGBoost (gain-based)
- LightGBM (split-based)
- Permutation Importance (model-agnostic, 15 repeats)
- Extra Trees (impurity-based)

**SHAP Deep Analysis:**
- LightGBM TreeExplainer for exact Shapley values
- Beeswarm plot (population-level — which features push predictions and in which direction)
- Bar summary (mean absolute SHAP — global feature ranking)

**8 Novel Clinically-Informed Features** (see [Feature Engineering section](#-feature-engineering))

**K-Means Clustering:** k=3 (determined by elbow method) → `cardiac_cluster` feature encoding latent patient sub-populations

</details>

<details>
<summary><b>📕 Phase 4 — Methodology & Results</b></summary>
<br/>

- **SMOTE + Tomek Links** applied to training set ONLY — prevents data leakage into test evaluation
- **5 base models** trained with 10-fold Stratified K-Fold CV
- **Optuna Bayesian HPO** for XGBoost and LightGBM: 50 trials each using Tree-structured Parzen Estimator (TPE)
- **Stacking Ensemble**: RF + Tuned XGB + Tuned LGBM + SVM → Logistic Regression meta-learner
  - Base learners generate out-of-fold predicted probabilities (5-fold CV)
  - Meta-learner trains on stacked probability matrix
- **6 evaluation metrics**: Accuracy, F1-Score, ROC-AUC, PR-AUC, Precision, Recall
- Confusion matrices, ROC curves, and calibration plots for all models

</details>

<details>
<summary><b>📒 Phase 5 — Research Comparison</b></summary>
<br/>

**5 Papers Reviewed (2022–2024):**
- Bharti et al. (2022) — Computational Intelligence and Neuroscience
- Ali et al. (2023) — IJACSA
- Mohan et al. (2023) — IEEE Access
- Louridi et al. (2023) — Health and Technology (Springer)
- Javeed et al. (2024) — Diagnostics (MDPI)

**Analysis Outputs:**
- 11-criterion comparison table across all 6 methodologies
- Radar chart: our work vs top 3 competitor papers across 6 dimensions
- Performance bar charts (Accuracy + ROC-AUC vs all papers)
- Detailed gap analysis with justification for each contribution
- LIME local explanations added as a Phase 5 improvement

</details>

---

## 🧠 Feature Engineering

We created **8 clinically-informed interaction features** — not arbitrary polynomial combinations, but features encoding specific pathophysiological relationships from cardiology literature:

| # | Feature | Formula | Clinical Meaning |
|---|---|---|---|
| 1 | `age_hr_ratio` | `age / thalach` | Cardiovascular efficiency — high age + low max HR = poor cardiac reserve |
| 2 | `bp_age_risk` | `trestbps × age / 1000` | Cumulative hypertensive load — longer BP exposure = more arterial damage |
| 3 | `chol_age_index` | `chol / age` | Age-adjusted cholesterol — same level is riskier in younger patients |
| 4 | `hr_reserve` | `(220 - age) - thalach` | Heart Rate Reserve gap — deficit from predicted maximum HR |
| 5 | `exercise_risk_score` | `oldpeak × (exang + 1)` | Compound ischemia score — ST depression magnitude × angina occurrence |
| 6 | `metabolic_risk_score` | `(chol/200) + (trestbps/120) + (fbs×0.5)` | Metabolic syndrome approximation |
| 7 | `vessel_thal_interaction` | `ca × thal` | Structural + functional cardiac risk product |
| 8 | `cp_exang_interaction` | `cp × (exang + 1)` | Symptom synergy — resting vs exertional pain pattern |

> 💡 **Validated impact:** `hr_reserve` ranked **#3 globally** by SHAP, `age_hr_ratio` ranked **#5** — confirming these features add real predictive signal. Engineering improved ROC-AUC by **+2.1%** over raw features alone.

---

## 🏆 Results

### Base Models — 10-Fold Cross-Validation

| Model | Accuracy | F1-Score | ROC-AUC | Precision | Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 84.8% | 85.7% | 0.913 | 85.1% | 86.3% |
| SVM (RBF) | 86.3% | 87.0% | 0.927 | 86.6% | 87.4% |
| Random Forest | 89.1% | 89.8% | 0.948 | 89.5% | 90.1% |
| LightGBM | 90.3% | 90.9% | 0.957 | 90.5% | 91.3% |
| XGBoost (Tuned) | 90.8% | 91.4% | 0.960 | 91.1% | 91.7% |
| LightGBM (Tuned) | 91.1% | 91.7% | 0.962 | 91.3% | 92.1% |

### Final Test Set Evaluation (Hold-Out 20%)

| Model | Accuracy | F1-Score | ROC-AUC | PR-AUC | Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| XGBoost (Tuned) | 91.8% | 92.3% | 0.963 | 0.946 | 92.5% |
| LightGBM (Tuned) | 91.3% | 91.9% | 0.961 | 0.943 | 92.1% |
| **🏆 Stacking Ensemble** | **93.2%** | **93.7%** | **0.965** | **0.951** | **94.0%** |

> ✅ The stacking ensemble outperforms every individual model across all metrics. The **94.0% Recall** is critical in a medical context — we correctly identify 94 out of every 100 true heart disease cases. Missing a true positive (false negative) in cardiac screening carries life-threatening consequences.

---

## 📈 Comparison with Related Work

| Paper | Dataset | Stacking | Feat. Eng. | Bayesian HPO | SHAP+LIME | Accuracy | AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Bharti et al. (2022) | 303 | ❌ | ❌ | ❌ | ❌ | 88.5% | — |
| Ali et al. (2023) | ~600 | ❌ | ❌ | ❌ | ❌ | 90.1% | 0.930 |
| Mohan et al. (2023) | 303 | ❌ | ❌ | ❌ | ❌ | 88.7% | — |
| Louridi et al. (2023) | ~800 | ❌ | PCA only | ❌ | Partial | 91.3% | 0.944 |
| Javeed et al. (2024) | 303 | ❌ | Basic bins | ❌ | Partial | 90.8% | 0.946 |
| **🏆 Our Work** | **920** | **✅** | **✅ 8 features** | **✅ Optuna** | **✅ Full** | **93.2%** | **0.965** |

---

## 🔍 Explainability — SHAP & LIME

A model that cannot explain its decisions **cannot be trusted in clinical settings**. We implement a full dual-layer explainability framework:

### 🌍 Global SHAP — Population-Level Insight
SHAP beeswarm plot reveals which features drive predictions across all patients, in what direction, and with what magnitude.

**Top 5 features by mean |SHAP|:**
1. `ca` — blocked vessels (strongest single predictor of structural CAD)
2. `thal` — reversible perfusion defects indicate active ischemia
3. `hr_reserve` — patients failing to reach predicted max HR carry elevated risk
4. `cp` — asymptomatic pain type paradoxically has highest disease association (silent ischemia)
5. `thalach` — low maximum heart rate reflects cardiac functional impairment

### 👤 Patient-Level SHAP Waterfall
For any individual patient, a waterfall plot shows the exact contribution of each feature to their specific risk score. Example high-risk patient (87% predicted probability):
- `ca = 2` → **+0.43** (two blocked vessels)
- `thal = 2` → **+0.31** (reversible defect)
- `hr_reserve = 41 bpm` → **+0.28** (large deficit from expected max HR)
- `cp = 0` → **+0.19** (asymptomatic pattern — silent ischemia)

### 🔎 LIME — Local Cross-Validation
LIME independently fits a local linear model around each individual prediction. When **SHAP and LIME agree** on feature direction → high confidence in the explanation. When they **disagree** → the case is borderline and warrants additional clinical investigation. This cross-method safety check is a novel feature of our pipeline.

---

## 🛠️ Tech Stack

```python
# Core
Python 3.10 · NumPy · Pandas

# Visualization  
Matplotlib · Seaborn · Plotly

# Machine Learning
Scikit-learn · XGBoost · LightGBM

# Explainability
SHAP (TreeExplainer) · LIME (LimeTabularExplainer)

# Optimization
Optuna (Bayesian HPO · TPE Sampler · 50 trials)

# Imbalance Handling
imbalanced-learn (SMOTE + Tomek Links)

# Platform
Kaggle Notebook (GPU Accelerated)
```

---

## 🚀 How to Run

### ▶️ Option 1 — Kaggle (Recommended)

1. Go to the [dataset page](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
2. Click **"New Notebook"** — the dataset auto-attaches at `/kaggle/input/`
3. Upload notebooks from the `notebooks/` folder **in order** (Phase 1 → Phase 5)
4. Run each notebook top-to-bottom with **Shift+Enter**
5. Each phase saves its output CSV for the next phase automatically

### 💻 Option 2 — Local Environment

```bash
# 1. Clone the repository
git clone https://github.com/azeemalley/Heart-Disease-ML-Project.git
cd Heart-Disease-ML-Project

# 2. Install all dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm \
            shap lime imbalanced-learn optuna plotly

# 3. Download dataset from Kaggle → save to data/

# 4. Run notebooks in sequence
jupyter notebook
# Open: Phase1 → Phase2 → Phase3 → Phase4 → Phase5
```

---

## 📄 Research Paper

A full **IEEE-style research paper** documenting all phases, methodology, results, and clinical insights is included:

📎 [`paper/Heart_Disease_Research_Paper.docx`](https://docs.google.com/document/d/1Ylp06qVlUzrjHiwWaJvxwqKlClT-MZSLRTLZ5L60Q5w/edit?usp=sharing)

| Section | Content |
|---|---|
| Abstract | Problem, method, results, and novelty summary |
| Introduction | CVD burden, clinical cost problem, 5 identified gaps |
| Related Work | Individual review of all 5 comparison papers with limitations |
| Dataset | Full feature table with clinical meanings, multi-center rationale |
| Methodology | All preprocessing, engineering, modeling, and explainability steps |
| Results | CV + test set tables, feature engineering impact, SHAP/LIME findings |
| Comparison | Full table vs 5 papers across 11 criteria |
| Conclusion | Findings summary + 3 future research directions |
| References | 12 properly formatted academic citations |

---

## 📊 Project Status

| Phase | Title | Status |
|---|---|---|
| Phase 1 | Problem Identification | ✅ Complete |
| Phase 2 | Visualization & Preprocessing | ✅ Complete |
| Phase 3 | Feature Engineering | ✅ Complete |
| Phase 4 | Methodology & Results | ✅ Complete |
| Phase 5 | Research Comparison | ✅ Complete |
| Phase 6 | Research Paper | ✅ Complete |

---

## 👤 Author

<div align="center">

**Muhammad Azeem**

BS Data Science — Machine Learning Course

[Punjab University College of Information and Technology] · Pakistan · 2026

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/azeemalley)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com/azeemalley)

</div>

---

<div align="center">

*"This is not just an academic exercise — it can save lives."*

<br/>

⭐ **If you found this project useful, please give it a star!**

</div>
