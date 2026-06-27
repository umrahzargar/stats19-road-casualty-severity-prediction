import json
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="STATS19 Road Casualty Severity Prediction",
    page_icon="🚦",
    layout="wide"
)


BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "stats19_xgb_pipeline.pkl"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"
SELECTED_FEATURES_PATH = BASE_DIR / "models" / "selected_features.json"

DEMO_DATA_PATH = BASE_DIR / "outputs" / "X_test_demo_sample.csv"
SHAP_GROUPED_PATH = BASE_DIR / "outputs" / "shap_grouped_feature_importance.csv"
OVERSAMPLING_PATH = BASE_DIR / "outputs" / "oversampling_sensitivity_summary.csv"
TEST_PREDICTIONS_PATH = BASE_DIR / "outputs" / "test_predictions.csv"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = cloudpickle.load(f)
    return model


@st.cache_data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def get_metric(metrics, possible_names, default=None):
    """
    Safely get a metric even if the JSON key has a slightly different name.
    """
    for name in possible_names:
        if name in metrics:
            return metrics[name]
    return default


def format_metric(value, decimals=3):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def predict_single_case(model, row, threshold):
    """
    Predict severe probability and class for one selected row.
    """
    probability = model.predict_proba(row)[0][1]
    predicted_class = 1 if probability >= threshold else 0
    return probability, predicted_class


try:
    model = load_model()
    metrics = load_json(METRICS_PATH)
    demo_data = load_csv(DEMO_DATA_PATH)
except Exception as e:
    st.error("The app could not load one or more required files.")
    st.exception(e)
    st.stop()


threshold = get_metric(
    metrics,
    possible_names=[
        "threshold",
        "final_threshold",
        "selected_threshold",
        "optimal_threshold",
        "best_threshold"
    ],
    default=0.5
)

try:
    threshold = float(threshold)
except Exception:
    threshold = 0.5


st.sidebar.title("🚦 STATS19 App")
st.sidebar.write("Road casualty severity prediction demo")

page = st.sidebar.radio(
    "Go to section:",
    [
        "Home / Project Summary",
        "Prediction Demo",
        "Model Performance",
        "SHAP Feature Importance",
        "Oversampling Sensitivity",
        "Limitations"
    ]
)

if page == "Home / Project Summary":
    st.title("🚦 Injury Severity Prediction for Road Casualties in Great Britain")

    st.markdown("""
    This Streamlit app presents a portfolio deployment of a machine learning project using
    **2023 Great Britain STATS19 road safety data**.

    The aim of the project is to predict whether a road casualty is likely to be:

    - **Severe**: Fatal or Serious injury  
    - **Non-severe**: Slight injury  

    The final model is a **cost-sensitive XGBoost classifier** trained to handle class imbalance.
    The model uses a selected set of casualty, vehicle, collision, and road-related features.
    """)

    st.subheader("Final Model")
    st.write("""
    The final model is a tuned XGBoost pipeline. It uses a selected decision threshold rather
    than the default 0.5 threshold, because the project focuses on improving detection of
    the minority Severe class.
    """)

    st.subheader("What this app shows")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("Prediction demo using saved test examples")

    with col2:
        st.info("Model performance metrics")

    with col3:
        st.info("SHAP-based feature importance")

    st.warning("""
    Disclaimer: This is a portfolio and academic demonstration only. 
    It should not be used for real legal, medical, emergency, policing, insurance, 
    or road safety decision-making.
    """)

elif page == "Prediction Demo":
    st.title("🔍 Prediction Demo")

    st.write("""
    Select one example casualty record from the demo dataset. 
    The saved XGBoost model will predict the probability that the case belongs to the Severe class.
    """)

    st.write(f"Demo dataset shape: **{demo_data.shape[0]} rows × {demo_data.shape[1]} columns**")

    row_number = st.selectbox(
        "Choose a demo sample:",
        options=list(range(len(demo_data))),
        format_func=lambda x: f"Sample {x}"
    )

    selected_row = demo_data.iloc[[row_number]]

    severe_probability, predicted_class = predict_single_case(
        model=model,
        row=selected_row,
        threshold=threshold
    )

    predicted_label = "Severe" if predicted_class == 1 else "Non-severe"

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Severe Probability", f"{severe_probability:.3f}")

    with col2:
        st.metric("Decision Threshold", f"{threshold:.3f}")

    with col3:
        st.metric("Predicted Class", predicted_label)

    if predicted_class == 1:
        st.error("The model predicts this case as Severe.")
    else:
        st.success("The model predicts this case as Non-severe.")

    st.subheader("Selected Input Features")
    st.dataframe(selected_row.T.rename(columns={row_number: "Value"}), use_container_width=True)


elif page == "Model Performance":
    st.title("📊 Model Performance")

    st.write("""
    These metrics summarise the final model performance on the test set.
    The project focuses especially on Severe-class recall because Severe casualties are the minority class.
    """)

    macro_f1 = get_metric(metrics, ["macro_f1", "macro-F1", "Macro-F1", "f1_macro"])
    severe_recall = get_metric(metrics, ["severe_recall", "Severe recall", "recall_severe"])
    severe_precision = get_metric(metrics, ["severe_precision", "Severe precision", "precision_severe"])
    roc_auc = get_metric(metrics, ["roc_auc", "ROC-AUC", "roc_auc_score"])
    avg_precision = get_metric(metrics, ["average_precision", "Average precision", "avg_precision"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Macro-F1", format_metric(macro_f1))
        st.metric("ROC-AUC", format_metric(roc_auc))

    with col2:
        st.metric("Severe Recall", format_metric(severe_recall))
        st.metric("Average Precision", format_metric(avg_precision))

    with col3:
        st.metric("Severe Precision", format_metric(severe_precision))
        st.metric("Selected Threshold", format_metric(threshold))

    st.subheader("Raw Metrics File")
    st.json(metrics)

    if TEST_PREDICTIONS_PATH.exists():
        st.subheader("Prediction Output Sample")
        predictions = load_csv(TEST_PREDICTIONS_PATH)
        st.dataframe(predictions.head(20), use_container_width=True)

elif page == "SHAP Feature Importance":
    st.title("🧠 SHAP Feature Importance")

    st.write("""
    SHAP was used to interpret the final model. 
    The chart below shows the most important grouped features based on the saved SHAP output.
    """)

    if SHAP_GROUPED_PATH.exists():
        shap_df = load_csv(SHAP_GROUPED_PATH)

        st.subheader("SHAP file preview")
        st.dataframe(shap_df.head(), use_container_width=True)

        # Try to automatically detect feature and importance columns
        possible_feature_cols = ["feature", "grouped_feature", "original_feature", "Feature"]
        possible_value_cols = [
            "mean_abs_shap",
            "mean_absolute_shap",
            "importance",
            "shap_importance",
            "Mean |SHAP|",
            "mean_abs_shap_value"
        ]

        feature_col = None
        value_col = None

        for col in possible_feature_cols:
            if col in shap_df.columns:
                feature_col = col
                break

        for col in possible_value_cols:
            if col in shap_df.columns:
                value_col = col
                break

        if feature_col is None:
            feature_col = shap_df.columns[0]

        if value_col is None:
            numeric_cols = shap_df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                value_col = numeric_cols[0]

        if feature_col is not None and value_col is not None:
            top_n = st.slider("Number of top features to show:", 5, 20, 10)

            top_features = (
                shap_df[[feature_col, value_col]]
                .sort_values(by=value_col, ascending=False)
                .head(top_n)
                .set_index(feature_col)
            )

            st.subheader(f"Top {top_n} SHAP Features")
            st.bar_chart(top_features)

            st.subheader("Top Features Table")
            st.dataframe(top_features, use_container_width=True)
        else:
            st.warning("Could not automatically identify SHAP feature and importance columns.")
    else:
        st.error("SHAP grouped feature importance file not found.")

elif page == "Oversampling Sensitivity":
    st.title("⚖️ Oversampling Sensitivity Analysis")

    st.write("""
    This section compares the final cost-sensitive XGBoost model with an oversampled XGBoost model.

    The oversampled version used RandomOverSampler inside an imblearn pipeline, while the final model
    used cost-sensitive learning through XGBoost.
    """)

    if OVERSAMPLING_PATH.exists():
        oversampling_df = load_csv(OVERSAMPLING_PATH)

        st.subheader("Oversampling Comparison")
        st.dataframe(oversampling_df, use_container_width=True)

        st.info("""
        In the dissertation/project workflow, RandomOverSampler did not improve Severe-class detection overall.
        Therefore, the cost-sensitive XGBoost model was kept as the final model.
        """)
    else:
        st.error("Oversampling sensitivity file not found.")


elif page == "Limitations":
    st.title("⚠️ Limitations")

    st.markdown("""
    This project has several important limitations:

    1. **2023 data only**  
       The model is based on one year of STATS19 data and may not generalise to other years.

    2. **Risk-screening only**  
       The model should be understood as a severity risk-screening tool, not a definitive classifier.

    3. **Class imbalance remains challenging**  
       Severe casualties are much less common than slight casualties, making prediction difficult.

    4. **Probability calibration not fully assessed**  
       The predicted probabilities should not be interpreted as perfectly calibrated real-world risk.

    5. **Spatial detail is limited**  
       More detailed road, junction, traffic, and location context could improve future modelling.

    6. **Portfolio demo only**  
       This app is not suitable for real-world legal, clinical, emergency, or insurance decisions.
    """)
