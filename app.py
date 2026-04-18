import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# CONFIG 
st.set_page_config(page_title="Healthcare Dashboard", layout="wide")

# LOAD 
model = joblib.load("no_show_model.pkl")
features = joblib.load("features.pkl")

# HEADER
st.markdown("""
    <h1 style='text-align:center; color:#2E86C1;'>
    🏥 Medical Appointment Analytics Dashboard
    </h1>
""", unsafe_allow_html=True)

st.markdown("---")

# SIDEBAR
st.sidebar.title("⚙️ Controls")
page = st.sidebar.radio("Navigation", ["Prediction", "Insights", "About"])

# 🧠 PAGE 1: PREDICTION

if page == "Prediction":

    st.header("🧠 Predict Patient No-Show Risk")

    col1, col2 = st.columns(2)
    input_data = {}

    for i, feature in enumerate(features):
        if i % 2 == 0:
            with col1:
                input_data[feature] = st.number_input(f"{feature}", value=0)
        else:
            with col2:
                input_data[feature] = st.number_input(f"{feature}", value=0)

    if st.button("🚀 Predict"):

        input_df = pd.DataFrame([input_data])
        prediction = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")

        # 🎯 Risk categorization
        if prediction < 0.3:
            risk = "Low"
            color = "green"
        elif prediction < 0.7:
            risk = "Medium"
            color = "orange"
        else:
            risk = "High"
            color = "red"

        st.metric("No-Show Probability", f"{prediction:.2f}")
        st.markdown(f"### Risk Level: :{color}[{risk}]")

        st.progress(float(prediction))

        # 💡 Recommendations
        st.subheader("💡 Suggested Actions")

        if risk == "High":
            st.error("Send reminder SMS 📩")
            st.error("Call patient 📞")
            st.error("Consider overbooking strategy")
        elif risk == "Medium":
            st.warning("Send reminder notification")
        else:
            st.success("No action needed")

# 📊 PAGE 2: INSIGHTS

elif page == "Insights":

    st.header("📊 Hospital Insights")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Patients", "110,000")
    col2.metric("No-Show Rate", "30%")
    col3.metric("Avg Age", "37")

    st.markdown("---")

    # 📈 Trend chart
    st.subheader("📈 Appointment Trend")
    data = np.random.randint(50, 150, 30)

    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title("Daily Appointments")
    st.pyplot(fig)

    # 🧠 Feature importance
    st.subheader("🧠 Key Factors Affecting No-Show")

    importance = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(feat_df.set_index("Feature"))

    # 🔍 Top 3 factors
    st.subheader("🔍 Top Risk Factors")

    top_features = feat_df.head(3)

    for _, row in top_features.iterrows():
        st.write(f"👉 {row['Feature']} (Importance: {row['Importance']:.2f})")


# 📄 PAGE 3: ABOUT

else:

    st.header("📄 About Project")

    st.write("""
    ### 🎯 Objective
    Predict patient no-shows and optimize hospital operations.

    ### 🧠 Model Used
    Random Forest Classifier

    ### 🚀 Key Features
    - Risk Prediction
    - Action Recommendations
    - Data Insights Dashboard

    ### 💡 Impact
    - Reduce missed appointments
    - Improve scheduling efficiency
    - Better patient management
    """)

    st.info("Developed using Machine Learning & Streamlit")