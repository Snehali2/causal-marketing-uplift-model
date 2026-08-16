import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="Marketing Campaign Analyzer",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# TITLE
# -----------------------------

st.title("📊 Marketing Campaign Effectiveness Analyzer")

st.write(
    "Causal uplift modeling using a T-Learner"
)

st.divider()


# -----------------------------
# FILE UPLOAD
# -----------------------------

st.subheader("1. Upload Marketing Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


# -----------------------------
# MODEL
# -----------------------------

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    # -------------------------
    # DATA PREVIEW
    # -------------------------

    st.subheader("2. Dataset Preview")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )

    # -------------------------
    # FEATURES
    # -------------------------

    features = [
        "age",
        "income",
        "previous_purchases",
        "website_visits"
    ]

    X = data[features]

    T = data["treatment"]

    Y = data["purchased"]

    # -------------------------
    # TREATMENT GROUP
    # -------------------------

    X_treatment = X[T == 1]

    Y_treatment = Y[T == 1]

    treatment_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    treatment_model.fit(
        X_treatment,
        Y_treatment
    )

    # -------------------------
    # CONTROL GROUP
    # -------------------------

    X_control = X[T == 0]

    Y_control = Y[T == 0]

    control_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    control_model.fit(
        X_control,
        Y_control
    )

    # -------------------------
    # PREDICTIONS
    # -------------------------

    data["prob_with_campaign"] = (
        treatment_model.predict_proba(X)[:, 1]
    )

    data["prob_without_campaign"] = (
        control_model.predict_proba(X)[:, 1]
    )

    # -------------------------
    # UPLIFT
    # -------------------------

    data["uplift"] = (
        data["prob_with_campaign"]
        -
        data["prob_without_campaign"]
    )

    # -------------------------
    # RECOMMENDATION
    # -------------------------

    data["recommendation"] = np.where(
        data["uplift"] > 0,
        "TARGET",
        "DO NOT TARGET"
    )

    # -------------------------
    # SUMMARY
    # -------------------------

    st.subheader("3. Model Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        len(data)
    )

    col2.metric(
        "Treatment Customers",
        int((T == 1).sum())
    )

    col3.metric(
        "Control Customers",
        int((T == 0).sum())
    )

    # -------------------------
    # RESULTS
    # -------------------------

    st.subheader("4. Customer Uplift Results")

    result_columns = [
        "customer_id",
        "prob_with_campaign",
        "prob_without_campaign",
        "uplift",
        "recommendation"
    ]

    results = (
        data[result_columns]
        .sort_values(
            "uplift",
            ascending=False
        )
    )

    st.dataframe(
        results.head(20),
        use_container_width=True
    )