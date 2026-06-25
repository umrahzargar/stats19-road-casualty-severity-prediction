# Injury Severity Prediction for Road Casualties in Great Britain

## Project Overview

This project applies machine learning to predict whether a road casualty outcome is **Severe** or **Slight** using the 2023 Great Britain STATS19 road safety dataset.

Severe casualties are defined as fatal or serious injuries, while slight injuries are treated as the non-severe class. The project uses collision, vehicle, and casualty-level administrative data to build a leakage-aware, interpretable classification pipeline.

The main goal is not only to predict severe casualty outcomes, but also to understand which road-user, vehicle, and collision-context features contribute most strongly to model predictions.

---

## Problem Statement

Road traffic collisions remain a major public safety concern. Predicting severe casualty risk from administrative collision records can support:

* post-collision risk screening
* road safety analysis
* identification of high-risk road-user groups
* evidence-informed transport safety planning

This project investigates whether STATS19 administrative records can distinguish Severe from Slight casualty outcomes and which features drive those predictions.

---

## Dataset

The project uses the 2023 Great Britain STATS19 road safety dataset.

Three linked tables were used:

* Collision table
* Vehicle table
* Casualty table

The final modelling dataset was created by merging the three tables at casualty level.

Dataset summary:

* 132,977 casualty-level records
* 42 selected raw predictors
* 190 processed features after preprocessing
* Binary target:

  * Severe = Fatal or Serious injury
  * Slight = non-severe injury

The target was imbalanced, with Severe casualties representing approximately 20.8% of records.
<img width="833" height="269" alt="image" src="https://github.com/user-attachments/assets/e4225d07-3683-4478-bc26-9a1b63527ee1" />

---

## Methodology

<img width="1492" height="1054" alt="image" src="https://github.com/user-attachments/assets/4dbce984-3507-4fe1-9c76-5f473e3d5a9d" />

### 1. Data Preparation

The casualty table was merged with the vehicle table using `collision_index` and `vehicle_reference`. Collision-level variables were then added using `collision_index`.

Severity-related variables were removed before modelling to reduce target leakage.

---

### 2. Feature Engineering

The final feature set covered six groups:

* Casualty characteristics
* Own vehicle and driver features
* Road infrastructure
* Environmental conditions
* Temporal features
* Collision-context aggregates

Temporal features were derived from date and time fields, while collision-context aggregates summarised the wider crash environment.

---

### 3. Leakage-Aware Splitting

All train/test and cross-validation splits were grouped by `collision_index`.

This was done because multiple casualties from the same collision can share road, vehicle, and environmental context. Grouped splitting reduced the risk of collision-level leakage between training and evaluation sets.

---

### 4. Preprocessing

Preprocessing was handled inside a Scikit-learn pipeline using a `ColumnTransformer`.

The pipeline included:

* median imputation for numeric variables
* standardisation for numeric variables
* missing-category imputation for categorical variables
* one-hot encoding for categorical variables

Keeping preprocessing inside the pipeline ensured that imputation, scaling, and encoding were fitted only on training folds during cross-validation.

---

### 5. Model Development

The following models were compared:

* Dummy majority-class baseline
* Logistic Regression
* Random Forest
* XGBoost

Class imbalance was handled using:

* balanced class weights for Logistic Regression and Random Forest
* `scale_pos_weight` for XGBoost

Hyperparameter tuning was performed using grouped cross-validation.

---

## Evaluation Metrics

The primary metric was **macro-F1**, because the target was imbalanced and macro-F1 gives equal weight to both classes.

Additional metrics included:

* G-Mean
* Balanced Accuracy
* Severe Recall
* Severe Precision
* Severe F1
* ROC-AUC
* Average Precision

A safety-aware threshold rule was used to avoid reducing Severe-class recall.

---

## Results

The final selected model was a tuned **XGBoost classifier**.

Held-out test performance:

| Metric            | Value |
| ----------------- | ----: |
| Macro-F1          | 0.619 |
| Severe Recall     | 0.649 |
| Severe Precision  | 0.361 |
| Severe F1         | 0.464 |
| G-Mean            | 0.670 |
| Balanced Accuracy | 0.670 |
| ROC-AUC           | 0.739 |
| Average Precision | 0.430 |

The final model correctly identified 3,673 of 5,658 Severe casualties.

The model is recall-oriented and is best interpreted as a **post-collision risk-screening tool**, not a definitive severity classifier.

<img width="401" height="243" alt="image" src="https://github.com/user-attachments/assets/6a2ef014-d21b-4c3f-9a2e-dfe0836e9fb2" />

<img width="466" height="335" alt="image" src="https://github.com/user-attachments/assets/16105548-fbad-4547-bc26-0e4b9d9d6a9a" />

<img width="469" height="331" alt="image" src="https://github.com/user-attachments/assets/f8f132d2-a7da-4a17-8059-227f2284b637" />

---

## Model Interpretation

SHAP analysis was used to explain the final XGBoost model.

The strongest grouped predictors were:

1. Casualty type
2. Vehicle manoeuvre
3. Vehicle type

These predictors highlight the value of merging casualty, vehicle, and collision-level data instead of relying only on collision-level information.

<img width="1067" height="827" alt="image" src="https://github.com/user-attachments/assets/1f39d1c1-fe56-44d7-8960-64bfbe265b1f" />
<img width="959" height="1127" alt="image" src="https://github.com/user-attachments/assets/25111d02-06ef-4c12-b54d-398db13d4482" />

---

## Oversampling Sensitivity Analysis

A supplementary experiment compared cost-sensitive XGBoost with RandomOverSampler.

Random oversampling did not improve Severe-class detection. It produced only a negligible macro-F1 gain while reducing Severe recall.

Cost-sensitive XGBoost was therefore retained as the final approach.

<img width="1187" height="587" alt="image" src="https://github.com/user-attachments/assets/4286a409-6b6d-4be7-9c9f-d72c1f103dc7" />

---

## Key Contributions

This project demonstrates:

* casualty-level modelling using all three STATS19 tables
* grouped validation by collision identifier to reduce data leakage
* comparison of linear and ensemble classifiers
* imbalance-aware model evaluation
* SHAP-based model interpretation
* safety-aware threshold selection
* realistic discussion of limitations and deployment context

---

## Repository Structure

```text
stats19-road-casualty-severity-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── stats19_road_casualty_severity_prediction.ipynb
│
├── reports/
│   └── stats19_road_casualty_severity_report.pdf
│
├── data/
│   └── README.md
│
└── images/
    └── README.md
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/umrahzargar/stats19-road-casualty-severity-prediction.git
```

2. Navigate to the project folder:

```bash
cd stats19-road-casualty-severity-prediction
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place the required STATS19 CSV files in the `data/` folder.

5. Open the notebook:

```bash
jupyter notebook notebooks/stats19_road_casualty_severity_prediction.ipynb
```

---

## Limitations and Future Work

The project uses only 2023 data, so temporal generalisation across multiple years was not tested. Probability calibration was not assessed, and the tuning budget was constrained by time and computational feasibility.

Future work could include:

* multi-year STATS19 modelling
* probability calibration
* wider hyperparameter search
* SMOTENC or oversampling before encoding
* spatial aggregation for geographically targeted road safety analysis
* interactive dashboard deployment using Streamlit or Power BI

---

## Author

**Umrah**

* GitHub: [github.com/umrahzargar](https://github.com/umrahzargar)
* LinkedIn: [linkedin.com/in/umrah-zargar](https://www.linkedin.com/in/umrah-zargar)
