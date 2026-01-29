import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "xgb_heart_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))

# ---------------- UI ----------------
st.set_page_config(
    page_title="Heart Attack Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Attack Risk Prediction Dashboard")
st.markdown("**AI-based Clinical Decision Support System (XGBoost + SHAP)**")

st.sidebar.header("🧍 Patient Input")

# ---------------- INPUT COLLECTION ----------------
input_dict = {}
for feature in features:
    input_dict[feature] = st.sidebar.number_input(
        label=feature,
        value=0.0,
        step=0.1
    )

input_df = pd.DataFrame([input_dict], columns=features)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Risk"):
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("🧾 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error(f"⚠️ **High Risk Detected**\n\nRisk Probability: **{probability*100:.2f}%**")
        else:
            st.success(f"✅ **Low Risk Detected**\n\nRisk Probability: **{probability*100:.2f}%**")

        st.progress(float(probability))

    # ---------------- FEATURE IMPORTANCE ----------------
    with col2:
        st.subheader("📊 Global Feature Importance")

        fi_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(fi_df.set_index("Feature"))

    # ---------------- SHAP EXPLANATION ----------------
    st.subheader("🧠 Local Explanation (Why this prediction?)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    shap_df = pd.DataFrame({
        "Feature": features,
        "SHAP Value": shap_values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    # Bar plot (Streamlit safe)
    fig, ax = plt.subplots()
    ax.barh(
        shap_df["Feature"][:10],
        shap_df["SHAP Value"][:10]
    )
    ax.set_title("Top Factors Influencing Prediction")
    ax.invert_yaxis()
    st.pyplot(fig)

    # ---------------- MEDICAL INTERPRETATION ----------------
    st.subheader("🩺 Medical Insight")

    if probability > 0.7:
        st.write("🔴 **High cardiovascular risk detected. Immediate medical consultation is advised.**")
    elif probability > 0.4:
        st.write("🟠 **Moderate risk detected. Lifestyle changes and regular monitoring recommended.**")
    else:
        st.write("🟢 **Low risk detected. Maintain a healthy lifestyle.**")

    # ---------------- DISCLAIMER ----------------
    st.warning(
        "⚠️ **Medical Disclaimer:** This AI system is for educational and support purposes only. "
        "It does NOT replace professional medical advice."
    )
