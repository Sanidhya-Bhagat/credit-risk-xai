import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Existing project components
# ---------------------------------------------------------

from src.config import DECISION_THRESHOLD
from src.metadata import MODEL_METADATA
from src.model_loader import load_xgboost_model
from src.risk import get_risk_band
from src.data import FEATURE_COLUMNS
from src.explain import create_shap_explainer
from src.lime_explain import create_lime_explainer
from src.split import create_train_test_split


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Credit Risk XAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Cached model loading
# ---------------------------------------------------------

@st.cache_resource
def get_model():
    """Load the existing persisted XGBoost model once."""
    return load_xgboost_model()

@st.cache_resource
def get_shap_explainer():
    return create_shap_explainer(get_model())

@st.cache_resource
def get_lime_explainer():
    X_train, _, y_train, _ = create_train_test_split()
    return create_lime_explainer(X_train, y_train)

@st.cache_data
def load_portfolio_risk_data():
    """Load the existing portfolio risk scores."""
    risk_path = PROJECT_ROOT / "outputs" / "risk_scores.csv"

    if not risk_path.exists():
        raise FileNotFoundError(
            f"Portfolio risk file not found: {risk_path}"
        )

    return pd.read_csv(risk_path)

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_probability(model, features):
    """Generate the model's default probability."""
    X = pd.DataFrame([features])

    probability = model.predict_proba(X)[0, 1]

    return float(probability)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📊 Credit Risk XAI")

st.subheader(
    "Explainable Credit Default Risk Assessment"
)

st.markdown(
    """
    An interactive machine-learning application for assessing
    credit-card default risk using an XGBoost model with
    SHAP and LIME explainability.
    """
)

st.caption(
    "Model: XGBoost  •  Decision Threshold: 0.30  •  "
    "Explainability: SHAP + LIME"
)

st.divider()

# ---------------------------------------------------------
# Navigation
# ---------------------------------------------------------

st.sidebar.title("Dashboard Menu")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Individual Risk Prediction",
        "Explainability",
        "Portfolio Risk Analysis",
        "Model & Threshold Analysis",
        "Calibration",
        "Model Limitations",
    ],
)

st.sidebar.caption(
    "Credit Risk XAI • XGBoost + SHAP + LIME"
)

st.sidebar.divider()

# ---------------------------------------------------------
# Overview
# ---------------------------------------------------------

if page == "Overview":

    st.header("Project Overview")

    st.markdown(
        """
        This dashboard provides an interactive interface for
        credit-risk prediction, risk classification, model
        explainability, and portfolio-level analysis.
        """
    )

    try:
        model = get_model()
        model_status = True
    except Exception as exc:
        model = None
        model_status = False
        st.error(f"Unable to load the XGBoost model: {exc}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model", "XGBoost")

    with col2:
        st.metric("ROC-AUC", "0.7775")

    with col3:
        st.metric("PR-AUC", "0.5552")

    with col4:
        st.metric(
            "Decision Threshold",
            f"{DECISION_THRESHOLD:.2f}",
        )

    st.subheader("Model Status")

    if model_status:
        st.success("XGBoost model loaded successfully.")

        status_col1, status_col2, status_col3 = st.columns(3)

        with status_col1:
            st.metric(
                "Expected Features",
                MODEL_METADATA["feature_count"],
            )

        with status_col2:
            st.metric(
                "Risk Bands",
                MODEL_METADATA["risk_band_count"],
            )

        with status_col3:
            st.metric(
                "Objective",
                MODEL_METADATA["objective"],
            )

    st.subheader("Dashboard Components")

    st.markdown(
        """
        - **Individual Risk Prediction** — assess an individual customer's
          probability of default.
        - **Explainability** — understand individual predictions using SHAP
          and LIME.
        - **Portfolio Risk Analysis** — explore the distribution of predicted
          risk across the test portfolio.
        - **Model & Threshold Analysis** — examine the business trade-off
          associated with the 0.30 decision threshold.
        - **Calibration** — review the model's probability behavior.
        - **Model Limitations** — understand important practical limitations.
        """
    )


# ---------------------------------------------------------
# Individual Risk Prediction
# ---------------------------------------------------------

elif page == "Individual Risk Prediction":

    st.header("Individual Credit Risk Prediction")

    st.markdown(
        """
        Enter the customer's financial and demographic information
        below to estimate their probability of credit-card default.
        """
    )

    # -----------------------------------------------------
    # Customer Profile
    # -----------------------------------------------------

    st.subheader("Customer Profile")

    profile_col1, profile_col2, profile_col3 = st.columns(3)

    with profile_col1:
        limit_bal = st.number_input(
            "Credit Limit (LIMIT_BAL)",
            min_value=0,
            value=50000,
            step=5000,
        )

    with profile_col2:
        sex_label = st.selectbox(
            "Sex",
            ["Male", "Female"],
        )

        sex = 1 if sex_label == "Male" else 2

    with profile_col3:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1,
        )

    profile_col4, profile_col5 = st.columns(2)

    with profile_col4:
        education_options = {
            "Graduate School": 1,
            "University": 2,
            "High School": 3,
            "Other": 4,
        }

        education_label = st.selectbox(
            "Education",
            list(education_options.keys()),
        )

        education = education_options[education_label]

    with profile_col5:
        marriage_options = {
            "Married": 1,
            "Single": 2,
            "Other": 3,
        }

        marriage_label = st.selectbox(
            "Marriage",
            list(marriage_options.keys()),
        )

        marriage = marriage_options[marriage_label]

    # -----------------------------------------------------
    # Repayment Status
    # -----------------------------------------------------

    st.subheader("Repayment Status")

    st.caption(
        "Repayment status values range from -2 to 8 according to "
        "the dataset's original encoding."
    )

    repayment_options = {
        "No consumption": -2,
        "Paid in full": -1,
        "Payment delayed 0 months": 0,
        "Payment delayed 1 month": 1,
        "Payment delayed 2 months": 2,
        "Payment delayed 3 months": 3,
        "Payment delayed 4 months": 4,
        "Payment delayed 5 months": 5,
        "Payment delayed 6 months": 6,
        "Payment delayed 7 months": 7,
        "Payment delayed 8+ months": 8,
    }

    repayment_labels = list(repayment_options.keys())

    pay_col1, pay_col2, pay_col3 = st.columns(3)

    with pay_col1:
        pay_0_label = st.selectbox(
            "September — PAY_0",
            repayment_labels,
        )

    with pay_col2:
        pay_2_label = st.selectbox(
            "August — PAY_2",
            repayment_labels,
        )

    with pay_col3:
        pay_3_label = st.selectbox(
            "July — PAY_3",
            repayment_labels,
        )

    pay_col4, pay_col5, pay_col6 = st.columns(3)

    with pay_col4:
        pay_4_label = st.selectbox(
            "June — PAY_4",
            repayment_labels,
        )

    with pay_col5:
        pay_5_label = st.selectbox(
            "May — PAY_5",
            repayment_labels,
        )

    with pay_col6:
        pay_6_label = st.selectbox(
            "April — PAY_6",
            repayment_labels,
        )

    pay_0 = repayment_options[pay_0_label]
    pay_2 = repayment_options[pay_2_label]
    pay_3 = repayment_options[pay_3_label]
    pay_4 = repayment_options[pay_4_label]
    pay_5 = repayment_options[pay_5_label]
    pay_6 = repayment_options[pay_6_label]

    # -----------------------------------------------------
    # Bill Statements
    # -----------------------------------------------------

    st.subheader("Bill Statements")

    st.caption(
        "Enter the customer's outstanding bill amount for each month."
    )

    bill_col1, bill_col2, bill_col3 = st.columns(3)

    with bill_col1:
        bill_amt1 = st.number_input(
            "September — BILL_AMT1",
            value=50000,
            step=1000,
        )

    with bill_col2:
        bill_amt2 = st.number_input(
            "August — BILL_AMT2",
            value=50000,
            step=1000,
        )

    with bill_col3:
        bill_amt3 = st.number_input(
            "July — BILL_AMT3",
            value=50000,
            step=1000,
        )

    bill_col4, bill_col5, bill_col6 = st.columns(3)

    with bill_col4:
        bill_amt4 = st.number_input(
            "June — BILL_AMT4",
            value=50000,
            step=1000,
        )

    with bill_col5:
        bill_amt5 = st.number_input(
            "May — BILL_AMT5",
            value=50000,
            step=1000,
        )

    with bill_col6:
        bill_amt6 = st.number_input(
            "April — BILL_AMT6",
            value=50000,
            step=1000,
        )

    # -----------------------------------------------------
    # Previous Payments
    # -----------------------------------------------------

    st.subheader("Previous Payments")

    st.caption(
        "Enter the amount paid by the customer for each month."
    )

    payment_col1, payment_col2, payment_col3 = st.columns(3)

    with payment_col1:
        pay_amt1 = st.number_input(
            "September — PAY_AMT1",
            min_value=0,
            value=2000,
            step=500,
        )

    with payment_col2:
        pay_amt2 = st.number_input(
            "August — PAY_AMT2",
            min_value=0,
            value=2000,
            step=500,
        )

    with payment_col3:
        pay_amt3 = st.number_input(
            "July — PAY_AMT3",
            min_value=0,
            value=2000,
            step=500,
        )

    payment_col4, payment_col5, payment_col6 = st.columns(3)

    with payment_col4:
        pay_amt4 = st.number_input(
            "June — PAY_AMT4",
            min_value=0,
            value=2000,
            step=500,
        )

    with payment_col5:
        pay_amt5 = st.number_input(
            "May — PAY_AMT5",
            min_value=0,
            value=2000,
            step=500,
        )

    with payment_col6:
        pay_amt6 = st.number_input(
            "April — PAY_AMT6",
            min_value=0,
            value=2000,
            step=500,
        )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    st.divider()

    assess = st.button(
        "Assess Credit Risk",
        type="primary",
        use_container_width=True,
    )

    if assess:

        features = {
            "LIMIT_BAL": limit_bal,
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
            "PAY_0": pay_0,
            "PAY_2": pay_2,
            "PAY_3": pay_3,
            "PAY_4": pay_4,
            "PAY_5": pay_5,
            "PAY_6": pay_6,
            "BILL_AMT1": bill_amt1,
            "BILL_AMT2": bill_amt2,
            "BILL_AMT3": bill_amt3,
            "BILL_AMT4": bill_amt4,
            "BILL_AMT5": bill_amt5,
            "BILL_AMT6": bill_amt6,
            "PAY_AMT1": pay_amt1,
            "PAY_AMT2": pay_amt2,
            "PAY_AMT3": pay_amt3,
            "PAY_AMT4": pay_amt4,
            "PAY_AMT5": pay_amt5,
            "PAY_AMT6": pay_amt6,
        }

        st.session_state["customer_features"] = features

        try:
            model = get_model()

            probability = get_probability(
                model,
                features,
            )

            st.session_state["prediction_probability"] = probability
            risk_band = get_risk_band(probability)

            st.session_state["risk_band"] = risk_band
            decision = probability >= DECISION_THRESHOLD

            st.session_state["decision"] = decision
            st.subheader("Credit Risk Assessment")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric(
                    "Default Probability",
                    f"{probability:.1%}",
                )

            with result_col2:
                st.metric(
                    "Risk Band",
                    risk_band,
                )

            with result_col3:
                if decision:
                    st.error("FLAGGED")
                else:
                    st.success("NOT FLAGGED")

            st.caption(
                f"Decision threshold: {DECISION_THRESHOLD:.0%}"
            )

        except Exception as exc:
            st.error(
                f"Unable to generate the prediction: {exc}"
            )


# ---------------------------------------------------------
# Explainability
# ---------------------------------------------------------

elif page == "Explainability":
    st.header("SHAP Explainability")
    st.write(
        "Understand which customer attributes are pushing the model "
        "toward or away from default."
    )

    if "customer_features" not in st.session_state:
        st.info(
            "Please complete an Individual Risk Prediction first "
            "to generate a SHAP explanation."
        )
    else:
        customer_features = st.session_state["customer_features"]

        if "prediction_probability" in st.session_state:
            probability = st.session_state["prediction_probability"]
        else:
            probability = None

        model = get_model()
        explainer = get_shap_explainer()

        X = pd.DataFrame(
            [customer_features],
            columns=FEATURE_COLUMNS,
        )

        explanation = explainer(X)
        shap_values = explanation.values[0]

        # Handle possible multi-output SHAP format
        if shap_values.ndim > 1:
            shap_values = shap_values[:, -1]

        shap_df = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Value": X.iloc[0].values,
                "SHAP Impact": shap_values,
            }
        )

        shap_df["Direction"] = shap_df["SHAP Impact"].apply(
            lambda value: "↑ Toward Default"
            if value > 0
            else "↓ Away from Default"
        )

        shap_df["Absolute Impact"] = shap_df["SHAP Impact"].abs()

        shap_df = shap_df.sort_values(
            "Absolute Impact",
            ascending=False,
        )

        st.subheader("Customer Risk Summary")

        if probability is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Default Probability",
                    f"{probability:.1%}",
                )

            with col2:
                st.metric(
                    "Risk Band",
                    get_risk_band(probability),
                )

        st.divider()

        st.subheader("Top Risk Drivers")

        top_features = shap_df.head(10).sort_values(
            "SHAP Impact",
            ascending=True,
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(
            top_features["Feature"],
            top_features["SHAP Impact"],
        )

        ax.axvline(
            0,
            linewidth=1,
        )

        ax.set_xlabel("SHAP Impact")
        ax.set_ylabel("Feature")
        ax.set_title("Top 10 Features Influencing This Prediction")

        st.pyplot(fig, width=800)

        st.caption(
            "Positive SHAP values push the model toward a higher default score; "
            "negative values push it toward a lower default score. "
            "SHAP impact is shown on the model's output scale and is not a "
            "percentage-point change in default probability."
        )

        st.subheader("Feature-Level Explanation")

        display_df = shap_df[
            [
                "Feature",
                "Value",
                "SHAP Impact",
                "Direction",
            ]
        ].copy()

        display_df["SHAP Impact"] = display_df["SHAP Impact"].round(4)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    # ---------------------------------------------------------
    # LIME Explainability
    # ---------------------------------------------------------

    st.divider()

    st.subheader("LIME Explainability")

    st.write(
        "Generate a local explanation showing which customer attributes "
        "are influencing this prediction."
    )

    if "customer_features" not in st.session_state:
        st.info(
            "Please complete an Individual Risk Prediction first "
            "to generate a LIME explanation."
        )
    else:
        if st.button("Generate LIME Explanation", type="secondary"):

            model = get_model()
            explainer = get_lime_explainer()

            customer_features = st.session_state["customer_features"]

            X_customer = pd.DataFrame(
                [customer_features],
                columns=FEATURE_COLUMNS,
            )

            customer = X_customer.iloc[0].values

            explanation = explainer.explain_instance(
                customer,
                model.predict_proba,
                num_features=10,
                num_samples=5000,
            )

            lime_results = explanation.as_list()

            lime_df = pd.DataFrame(
                lime_results,
                columns=["Feature Condition", "Contribution"],
            )

            lime_df["Direction"] = lime_df["Contribution"].apply(
                lambda value: "↑ Toward Default"
                if value > 0
                else "↓ Away from Default"
            )

            lime_df["Contribution"] = lime_df["Contribution"].round(4)

            st.subheader("Top LIME Drivers")

            st.dataframe(
                lime_df,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Positive contributions push the explanation toward "
                "the Default class; negative contributions push it "
                "toward the No Default class."
            )



# ---------------------------------------------------------
# Portfolio Risk Analysis
# ---------------------------------------------------------

elif page == "Portfolio Risk Analysis":

    st.header("Portfolio Risk Analysis")

    try:
        portfolio_df = load_portfolio_risk_data()

        total_customers = len(portfolio_df)
        average_probability = portfolio_df["predicted_probability"].mean()
        flagged_customers = portfolio_df["decision_at_0_30"].sum()
        flagged_percentage = flagged_customers / total_customers

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric(
                "Total Customers",
                f"{total_customers:,}",
            )

        with metric_col2:
            st.metric(
                "Average Default Probability",
                f"{average_probability:.1%}",
            )

        with metric_col3:
            st.metric(
                "Flagged Customers",
                f"{flagged_customers:,}",
            )

        with metric_col4:
            st.metric(
                "Flagged Percentage",
                f"{flagged_percentage:.1%}",
            )

    except Exception as exc:
        st.error(
            f"Unable to load portfolio risk data: {exc}"
        )

    st.divider()

    st.markdown(
    """
    <style>
    .stMainBlockContainer img,
    .block-container img,
    section[data-testid="stMain"] img {
        max-width: 800px !important;
        width: 100% !important;
        height: auto !important;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

    st.subheader("Risk-Band Distribution")

    risk_band_order = [
        "0–10%",
        "10–20%",
        "20–30%",
        "30–40%",
        "40–50%",
        "50–60%",
        "60–70%",
        "70–80%",
        "80–90%",
        "90–100%",
    ]

    risk_band_counts = (
        portfolio_df["risk_band"]
        .value_counts()
        .reindex(risk_band_order, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10, 5), dpi=72)

    ax.bar(
        risk_band_counts.index,
        risk_band_counts.values,
    )

    ax.set_xlabel("Risk Band")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Portfolio Distribution by Predicted Risk Band")

    plt.xticks(rotation=45)

    st.pyplot(fig, width="stretch")

    st.divider()

    st.subheader("Decision Distribution")

    decision_counts = (
        portfolio_df["decision_at_0_30"]
        .map({
            0: "Not Flagged",
            1: "Flagged",
        })
        .value_counts()
        .reindex(
            ["Not Flagged", "Flagged"],
            fill_value=0,
        )
    )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=72)

    ax.bar(
        decision_counts.index,
        decision_counts.values,
    )

    ax.set_xlabel("Decision")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Portfolio Distribution by Credit-Risk Decision")

    st.pyplot(fig, width="stretch")

# ---------------------------------------------------------
# Model & Threshold Analysis
# ---------------------------------------------------------

elif page == "Model & Threshold Analysis":

    st.header("Model & Threshold Analysis")

    st.markdown(
        """
        Compare the validated decision threshold of **0.30** with
        the default classification threshold of **0.50**.
        """
    )

    st.subheader("Threshold Performance Comparison")

    threshold_df = pd.DataFrame(
        {
            "Decision Threshold": ["0.50", "0.30"],
            "Precision": [0.6694, 0.5400],
            "Recall": [0.3632, 0.5335],
            "F1 Score": [0.4709, 0.5368],
        }
    )

    st.dataframe(
        threshold_df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Business Trade-Off")

    tradeoff_col1, tradeoff_col2 = st.columns(2)

    with tradeoff_col1:
        st.metric(
            "Recall Improvement",
            f"{(0.5335 - 0.3632):.1%}",
        )

    with tradeoff_col2:
        st.metric(
            "F1 Improvement",
            f"{(0.5368 - 0.4709):.1%}",
        )

    st.info(
        "The 0.30 threshold increases recall and F1 score compared "
        "with 0.50, meaning the model identifies more default cases "
        "while accepting a reduction in precision. The 0.30 threshold "
        "is therefore retained as the project's validated decision "
        "threshold."
    )


# ---------------------------------------------------------
# Calibration
# ---------------------------------------------------------

elif page == "Calibration":

    st.header("Calibration Analysis")

    st.markdown(
        """
        Calibration measures how closely the model's predicted
        probabilities correspond to observed default outcomes.
        """
    )

    st.subheader("Calibration Performance")

    calibration_col1, calibration_col2 = st.columns(2)

    with calibration_col1:
        st.metric(
            "Brier Score",
            "0.1354",
        )

    with calibration_col2:
        st.metric(
            "Calibration Adjustment",
            "Not Applied",
        )

    st.divider()

    st.subheader("Calibration Findings")

    st.markdown(
        """
        - **Below approximately 60%:** predicted probabilities show
          relatively strong alignment with observed outcomes.
        - **Approximately 60–80%:** the model shows signs of
          overconfidence in its predicted probabilities.
        - **90–100%:** this range contains only six observations and
          should therefore be treated as unstable.
        """
    )

    st.info(
        "The model has not been recalibrated. Predicted probabilities "
        "should therefore not be interpreted as perfectly calibrated "
        "estimates of default risk."
    )


# ---------------------------------------------------------
# Model Limitations
# ---------------------------------------------------------

elif page == "Model Limitations":

    st.header("Model Limitations")

    st.markdown(
        """
        ### Important considerations

        - Model performance depends on the characteristics of the
          underlying dataset.
        - Predicted probabilities should not be interpreted as perfectly
          calibrated probabilities.
        - Model performance may change under distribution shift.
        - The 0.30 decision threshold represents a trade-off between
          recall and precision.
        - SHAP and LIME provide explanations of model behavior but do
          not establish causal relationships.
        - The model should support responsible credit decisions rather
          than replace appropriate human and institutional oversight.
        """
    )
