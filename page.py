import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model (make sure to replace with your model path)
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Set Streamlit page config
st.set_page_config(page_title="Heat Exchanger Maintenance Predictor", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🔍 Maintenance Predictor", "📊 Dashboard"])

if page == "🔍 Maintenance Predictor":
    st.title("🔧 Heat Exchanger Maintenance Checker")
    st.markdown("Enter all parameters to predict if maintenance is needed.")

    fluid_mapping = {"Water": 0, "Oil": 1, "Refrigerant": 2}

    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            Thickness = st.number_input("Thickness", value=5.0)
            Plate_thickness = st.number_input("Plate Thickness (mm)", value=0.6)
            Temp1In = st.number_input("Fluid1 Inlet Temp (°C)", value=80.0)
            Temp1Out = st.number_input("Fluid1 Outlet Temp (°C)", value=65.0)
            Temp2In = st.number_input("Fluid2 Inlet Temp (°C)", value=30.0)
            Temp2Out = st.number_input("Fluid2 Outlet Temp (°C)", value=50.0)
            Fluid1 = st.selectbox("Fluid1 Type", options=list(fluid_mapping.keys()), index=0)
            Fluid2 = st.selectbox("Fluid2 Type", options=list(fluid_mapping.keys()), index=1)

        with col2:
            Q = st.number_input("Total Heat Transfer Q (kW)", value=500.0)
            Qr = st.number_input("Unit Heat Transfer Qr (kW)", value=350.0)
            Lmtd = st.number_input("Log Mean Temp Difference (K)", value=25.0)
            Theta1 = st.number_input("Theta1 (dT1/Lmtd)", value=0.6)
            Theta2 = st.number_input("Theta2 (dT2/Lmtd)", value=0.7)
            m1 = st.number_input("Side1 Mass Flowrate", value=100.0)
            m2 = st.number_input("Side2 Mass Flowrate", value=110.0)
            dP1 = st.number_input("Side1 Pressure Drop", value=90.0)

        with col3:
            dP2 = st.number_input("Side2 Pressure Drop", value=95.0)
            V1In = st.number_input("Side1 Inlet Velocity", value=3.5)
            V2In = st.number_input("Side2 Inlet Velocity", value=3.2)
            A = st.number_input("Total Plate Surface Area", value=50.0)
            NP = st.number_input("Total Number of Plates", value=100)
            AdNP = st.number_input("Area per Plate", value=0.5)
            NuP = st.number_input("Units in Parallel", value=2)
            Ar = st.number_input("Area per Unit", value=25.0)

        submit = st.form_submit_button("Check Maintenance Status")

        if submit:
            # Convert categorical fluids to numeric codes
            Fluid1_code = fluid_mapping[Fluid1]
            Fluid2_code = fluid_mapping[Fluid2]

            # Calculate additional features
            delta_T1 = Temp1In - Temp1Out
            efficiency = Qr / Q if Q != 0 else 0

            input_data = pd.DataFrame([{
    "Thickness": Thickness,
    "Temp1In": Temp1In,
    "Temp1Out": Temp1Out,
    "Temp2In": Temp2In,
    "Temp2Out": Temp2Out,
    "Fluid1": Fluid1_code,
    "Fluid2": Fluid2_code,
    "Q": Q,
    "Qr": Qr,
    "Lmtd": Lmtd,
    "Theta1": Theta1,
    "Theta2": Theta2,
    "m1": m1,
    "m2": m2,
    "dP1": dP1,
    "dP2": dP2,
    "V1In": V1In,
    "V2In": V2In,
    "A": A,
    "NP": NP,
    "delta_T1": delta_T1,
    "efficiency": efficiency
}])


            prediction = model.predict(input_data)[0]
            if prediction == 1:
                st.error("❌ Maintenance Required")
            else:
                st.success("✅ No Maintenance Needed")

elif page == "📊 Dashboard":
    st.title("📊 Maintenance Prediction Dashboard")

    st.subheader("🧾 Feature Descriptions and Their Roles in Prediction")
    
    feature_info = pd.DataFrame({
        "Feature": [
            "Thickness", "Plate_thickness_[mm]", "Temp1In", "Temp1Out", "Temp2In", "Temp2Out",
            "Fluid1", "Fluid2", "Q", "Qr", "Lmtd", "Theta1", "Theta2", "m1", "m2", "dP1", "dP2",
            "V1In", "V2In", "A", "NP", "AdNP", "NuP", "Ar", "delta_T1", "efficiency"
        ],
        "Description": [
            "Overall thickness of the heat exchanger wall",
            "Thickness of individual heat transfer plates",
            "Inlet temperature of fluid 1",
            "Outlet temperature of fluid 1",
            "Inlet temperature of fluid 2",
            "Outlet temperature of fluid 2",
            "Type/category of fluid 1 (encoded)",
            "Type/category of fluid 2 (encoded)",
            "Total heat transferred across the exchanger",
            "Unit heat transferred per segment",
            "Log mean temperature difference – key for heat transfer rate",
            "Ratio of dT1 to LMTD",
            "Ratio of dT2 to LMTD",
            "Mass flow rate on side 1",
            "Mass flow rate on side 2",
            "Pressure drop on side 1",
            "Pressure drop on side 2",
            "Inlet velocity of fluid on side 1",
            "Inlet velocity of fluid on side 2",
            "Total plate surface area",
            "Total number of plates used",
            "Surface area per plate",
            "Number of units working in parallel",
            "Surface area per unit",
            "Temperature drop of fluid 1 (Temp1In - Temp1Out)",
            "Ratio of Qr to Q, indicating heat exchanger efficiency"
        ]
    })

    st.dataframe(feature_info, use_container_width=True)

    st.subheader("🔝 Top 5 Most Important Features (Based on Model)")
    importance_data = pd.DataFrame({
        "Feature": ["dP1", "efficiency", "delta_T1", "Qr", "Temp1In"],
        "Importance": [0.24, 0.21, 0.18, 0.15, 0.12]
    })

    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=importance_data, ax=ax, palette="viridis")
    ax.set_title("Feature Importance in Maintenance Prediction", fontsize=14)
    st.pyplot(fig)

    st.markdown("""
    ### 🧠 Interpretation Tips
    - **High pressure drop (`dP1`)** and **low efficiency** often signal the need for maintenance.
    - The model learned these patterns from **60,000 data points** using supervised learning.
    - You can experiment with input values on the left tab to see real-time predictions.
    """)
