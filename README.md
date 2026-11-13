#  Disease Prediction from Medical Data — CodeAlpha ML Internship (Task 4)

Predicting whether a breast tumor is **malignant** or **benign** from clinical measurements, using and comparing four classification algorithms.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m-9_wnLenvWJBUTJw04hM9Vjss9djwko)

##  Objective
Build a model that predicts the possibility of disease from structured patient data, with a focus on **not missing malignant cases** (high recall).

##  Dataset
**Breast Cancer Wisconsin (Diagnostic)** — UCI Machine Learning Repository (bundled in scikit-learn).
- 569 patients, 30 numerical features computed from digitized biopsy images (radius, texture, perimeter, area, concavity, …)
- Target: `1` = malignant (37%), `0` = benign (63%)

##  Approach
1. **Exploratory Data Analysis**: class balance, correlations, feature distributions
2. **Preprocessing**: stratified 80/20 split, `StandardScaler` inside scikit-learn `Pipeline`s (no data leakage)
3. **Model comparison** with 5-fold stratified cross-validation:
   Logistic Regression · SVM · Random Forest · XGBoost
4. **Hyperparameter tuning** of the best model with `GridSearchCV` (F1 score)
5. **Evaluation** on a held-out test set: Precision, Recall, F1, ROC-AUC, confusion matrix, ROC curves
6. **Decision threshold analysis** to trade false alarms for fewer missed cancers
7. **Feature importance** analysis

##  Results

Cross-validation (train set, 5 folds):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.974** | **0.977** | **0.953** | **0.964** | **0.996** |
| XGBoost | 0.971 | 0.970 | 0.953 | 0.961 | 0.992 |
| SVM | 0.971 | 0.977 | 0.947 | 0.961 | 0.995 |
| Random Forest | 0.960 | 0.958 | 0.935 | 0.946 | 0.988 |

Final model (**Logistic Regression**) on the test set (114 patients):

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.965 | 0.975 | 0.929 | 0.951 | **0.996** |

Lowering the decision threshold from 0.5 to **0.3** raises recall to **0.976** (only 1 missed cancer out of 42) while keeping precision at 0.976.

**Most predictive features:** worst area, worst perimeter, worst radius and worst concave points (tumor size and contour irregularity).

##  How to run
**Google Colab (recommended):** click the badge above, then *Runtime → Run all*.

**Locally:**
```bash
git clone https://github.com/inesbl1/CodeAlpha_DiseasePrediction.git
cd CodeAlpha_DiseasePrediction
pip install -r requirements.txt
jupyter notebook Disease_Prediction.ipynb
```

##  Tech stack
Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn · joblib

##  Future work
- Apply the same pipeline to the Diabetes (Pima) and Heart Disease datasets
- Explain individual predictions with SHAP
- Deploy an interactive app with Streamlit

>  Educational project — not a substitute for medical diagnosis.

---
Made by **Ines BOUALI** during the **CodeAlpha Machine Learning Internship** · [LinkedIn](https://www.linkedin.com/in/ines-bouali-6b4b69311)