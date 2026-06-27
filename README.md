# Injury Severity Prediction for Road Casualties in Great Britain

## Live Streamlit App

**Try the deployed app here:**
https://stats19-road-casualty-severity-prediction.streamlit.app/

The Streamlit app allows users to:

* view the project summary and model purpose
* test single-case predictions using built-in demo samples
* upload a preprocessed STATS19-style file for batch prediction
* download an example input template
* generate Severe-class probabilities and predicted severity labels
* filter predictions by probability threshold
* download full prediction results
* view model performance metrics
* explore SHAP-based feature importance
* review the oversampling sensitivity analysis

> **Important:** The app is an academic and portfolio demonstration only. It should not be used for real legal, medical, emergency, policing, insurance, or road safety decision-making.

---

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

The project uses the **2023 Great Britain STATS19 road safety dataset**.

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
  * Slight = Non-severe injury

The target was imbalanced, with Severe casualties representing approximately **20.8%** of records.

<img width="833" height="269" alt="Target distribution" src="https://github.com/user-attachments/assets/e4225d07-3683-4478-bc26-9a1b63527ee1" />

---

## Methodology

<img width="1492" height="1054" alt="Project methodology" src="https://github.com/user-attachments/assets/4dbce984-3507-4fe1-9c76-5f473e3d5a9d" />

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

The final selected model was a tuned **cost-sensitive XGBoost classifier**.

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

The final model correctly identified **3,673 of 5,658 Severe casualties**.

The model is recall-oriented and is best interpreted as a **post-collision risk-screening tool**, not a definitive severity classifier.

<img width="401" height="243" alt="Model result chart" src="https://github.com/user-attachments/assets/6a2ef014-d21b-4c3f-9a2e-dfe0836e9fb2" />

<img width="466" height="335" alt="Model performance chart" src="https://github.com/user-attachments/assets/16105548-fbad-4547-bc26-0e4b9d9d6a9a" />

<img width="469" height="331" alt="Model evaluation chart" src="https://github.com/user-attachments/assets/f8f132d2-a7da-4a17-8059-227f2284b637" />

---

## Streamlit Deployment

The final model was deployed using **Streamlit Community Cloud**.

The app provides an interactive interface for exploring the trained model.

### App Features

#### 1. Project Summary

The home page explains the dataset, modelling task, target definition, final model, and responsible-use disclaimer.

#### 2. Single-Case Prediction Demo

Users can select a saved demo sample from the test set. The app returns:

* Severe probability
* selected decision threshold
* predicted class
* input feature values for the selected case

#### 3. Batch Upload Prediction

Users can upload a preprocessed STATS19-style file with the same feature columns as the model input.

The app then:

* checks whether required columns are present
* ignores extra columns if present
* predicts Severe probability for all rows
* applies the selected classification threshold
* shows a prediction summary
* displays a risk distribution chart
* allows probability-threshold filtering
* shows highest-risk cases
* allows users to download prediction results as CSV

#### 4. Model Performance Page

The app displays the main model evaluation metrics, including Macro-F1, Severe recall, Severe precision, ROC-AUC, Average Precision, and the selected threshold.

#### 5. SHAP Feature Importance Page

The app displays grouped SHAP feature importance to support model interpretability.

#### 6. Oversampling Sensitivity Page

The app compares the final cost-sensitive XGBoost model against an oversampled XGBoost model using RandomOverSampler.

---

## Model Interpretation

SHAP analysis was used to explain the final XGBoost model.

The strongest grouped predictors were:

1. Casualty type
2. Vehicle manoeuvre
3. Vehicle type

These predictors highlight the value of merging casualty, vehicle, and collision-level data instead of relying only on collision-level information.

<img width="1067" height="827" alt="SHAP grouped feature importance" src="https://github.com/user-attachments/assets/1f39d1c1-fe56-44d7-8960-64bfbe265b1f" />

<img width="959" height="1127" alt="SHAP feature interpretation" src="https://github.com/user-attachments/assets/25111d02-06ef-4c12-b54d-398db13d4482" />

---

## Oversampling Sensitivity Analysis

A supplementary experiment compared cost-sensitive XGBoost with RandomOverSampler.

Random oversampling did not improve Severe-class detection overall. It produced only a negligible macro-F1 gain while reducing Severe recall.

Cost-sensitive XGBoost was therefore retained as the final approach.

<img width="1187" height="587" alt="Oversampling sensitivity comparison" src="https://github.com/user-attachments/assets/4286a409-6b6d-4be7-9c9f-d72c1f103dc7" />

---

## Key Contributions

This project demonstrates:

* casualty-level modelling using all three STATS19 tables
* grouped validation by collision identifier to reduce data leakage
* comparison of linear and ensemble classifiers
* imbalance-aware model evaluation
* safety-aware threshold selection
* SHAP-based model interpretation
* oversampling sensitivity analysis
* interactive Streamlit deployment
* batch prediction using uploaded STATS19-style files

---

## Repository Structure

```text
stats19-road-casualty-severity-prediction/
│
├── README.md
├── app.py
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── stats19_xgb_pipeline.pkl
│   ├── model_metrics.json
│   ├── selected_features.json
│   ├── shap_values.pkl
│   ├── shap_feature_names.pkl
│   ├── X_shap_raw.pkl
│   ├── X_shap_processed_dense.pkl
│   └── shap_sample.pkl
│
├── outputs/
│   ├── X_test.xls
│   ├── y_test.xls
│   ├── X_test_demo_sample.xls
│   ├── X_test_sample.xls
│   ├── test_predictions.xls
│   ├── shap_grouped_feature_importance.xls
│   ├── shap_transformed_feature_importance.xls
│   └── oversampling_sensitivity_summary.xls
│
├── Notebook/
│   └── project notebook files
│
└── Report/
    └── project report files
```

---

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/umrahzargar/stats19-road-casualty-severity-prediction.git
```

2. Navigate to the project folder:

```bash
cd stats19-road-casualty-severity-prediction
```

3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the virtual environment.

On Windows Command Prompt:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Run the Streamlit app:

```bash
streamlit run app.py
```

The app should open locally at:

```text
http://localhost:8501
```

---

## Requirements

The main Python libraries used are:

* streamlit
* pandas
* numpy
* scikit-learn
* xgboost
* cloudpickle
* shap
* matplotlib
* openpyxl
* xlrd

---

## Deployment Notes

The model was saved using `cloudpickle` because standard `joblib` saving caused a pickling issue with a lambda function inside the pipeline.

The Streamlit app therefore loads the model using:

```python
import cloudpickle

with open("models/stats19_xgb_pipeline.pkl", "rb") as f:
    model = cloudpickle.load(f)
```

A small compatibility fix was also added to support loading the saved Scikit-learn pipeline across deployment environments.

---

## Limitations and Future Work

The project has several important limitations:

* The model uses only 2023 data, so temporal generalisation across multiple years has not been fully tested.
* Uploaded files must already be preprocessed into the same feature format as the demo input file.
* Probability calibration was not fully assessed.
* The model is recall-oriented and should be interpreted as a screening tool, not a definitive classifier.
* Spatial detail is limited.
* The tuning budget was constrained by time and computational feasibility.

Future work could include:

* multi-year STATS19 modelling
* probability calibration
* wider hyperparameter search
* SMOTENC or oversampling before encoding
* spatial aggregation for geographically targeted road safety analysis
* automated preprocessing for raw STATS19 files from other years
* richer dashboard features such as maps, temporal trends, and subgroup analysis

---

## Responsible Use

This project is intended for academic learning, portfolio demonstration, and exploratory road safety analysis.

It should **not** be used for real-world legal, medical, emergency, policing, insurance, or operational road safety decision-making.

---

## Author

**Umrah**

* GitHub: [github.com/umrahzargar](https://github.com/umrahzargar)
* LinkedIn: [linkedin.com/in/umrah-zargar](https://www.linkedin.com/in/umrah-zargar)
