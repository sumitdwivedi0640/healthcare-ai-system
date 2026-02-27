import streamlit as st
import requests
import os
import pandas as pd

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Healthcare AI System",
    page_icon="🏥",
    layout="wide"
)

API_URL = "http://localhost:8000"
API_KEY = "healthcare-secret-key"

headers = {
    "x-api-key": API_KEY
}

st.title("🏥 Healthcare AI System")

menu = st.sidebar.selectbox(
    "Select Module",
    [
        "Sepsis Prediction",
        "Tumor Detection",
        "Treatment Recommendation",
        "View History"
    ]
)

# =========================================================
# SEPSIS MODULE
# =========================================================
if menu == "Sepsis Prediction":

    st.header("🦠 Sepsis Prediction")

    patient_name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    features = st.text_area(
        "Enter 43 comma-separated values",
        "80,120,98,6,95,7,110,85,70,1,0,36.5,102,18,0.5,1.2,140,4.5,13,250,0,1,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0"
    )

    if st.button("Predict Sepsis"):

        try:
            feature_list = [float(x.strip()) for x in features.split(",")]

            response = requests.post(
                f"{API_URL}/predict-sepsis",
                headers=headers,
                json={
                    "patient_name": patient_name,
                    "age": age,
                    "gender": gender,
                    "features": feature_list
                }
            )

            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                probability = result["probability"]

                col1, col2 = st.columns(2)

                with col1:
                    if prediction == 1:
                        st.error("🚨 High Risk of Sepsis")
                    else:
                        st.success("✅ Low Risk of Sepsis")

                with col2:
                    st.metric(
                        label="Model Confidence",
                        value=f"{round(probability*100, 2)}%"
                    )

                st.progress(min(probability, 1.0))

            else:
                st.error(response.text)

        except Exception as e:
            st.error(f"Error: {e}")

# =========================================================
# TUMOR MODULE
# =========================================================
elif menu == "Tumor Detection":

    st.header("🧠 Tumor Detection")

    patient_name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    uploaded_file = st.file_uploader(
        "Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        os.makedirs("uploads", exist_ok=True)
        save_path = os.path.join("uploads", uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        if st.button("Detect Tumor"):

            response = requests.post(
                f"{API_URL}/predict-tumor",
                headers=headers,
                json={
                    "patient_name": patient_name,
                    "age": age,
                    "gender": gender,
                    "image_path": save_path
                }
            )
            
            if response.status_code == 200:
                result = response.json()

    # Show detected modality
    if "detected_modality" in result:
        st.info(f"Detected Modality: {result['detected_modality']}")

    # If tumor prediction exists
    if "label" in result:
        label = result["label"]
        confidence = result["confidence"]

        col1, col2 = st.columns(2)

        with col1:
            if label.lower() == "tumor detected":
                st.error("🧠 Tumor Detected")
            else:
                st.success("✅ No Tumor Detected")

        with col2:
            st.metric(
                label="Model Confidence",
                value=f"{round(confidence*100, 2)}%"
            )

        st.progress(min(confidence, 1.0))

    # If MRI not detected
    elif "message" in result:
        st.warning(result["message"])
# =========================================================
# TREATMENT MODULE
# =========================================================
elif menu == "Treatment Recommendation":

    st.header("💊 Treatment Recommendation")

    patient_name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    disease = st.selectbox("Disease", ["sepsis", "tumor"])
    severity = st.selectbox("Severity", ["low", "medium", "high"])

    if st.button("Get Recommendation"):

        response = requests.post(
            f"{API_URL}/recommend-treatment",
            headers=headers,
            json={
                "patient_name": patient_name,
                "age": age,
                "gender": gender,
                "disease": disease,
                "severity": severity
            }
        )

        if response.status_code == 200:

            result = response.json()

            st.subheader("Recommended Action")
            st.info(result["recommended_action"])

            st.subheader("Treatment Plan")
            st.write(result["treatment_plan"])

            if result["urgency"].lower() == "critical":
                st.error("⚠️ CRITICAL CASE")
            elif result["urgency"].lower() == "high":
                st.warning("⚠️ High Risk")
            else:
                st.success("Stable Condition")

        else:
            st.error(response.text)

# =========================================================
# HISTORY VIEWER
# =========================================================
elif menu == "View History":

    st.header("📜 Prediction History")

    history_type = st.selectbox(
        "Select History Type",
        ["Sepsis", "Tumor", "Treatment"]
    )

    if st.button("Load History"):

        if history_type == "Sepsis":
            response = requests.get(
                f"{API_URL}/history/sepsis", headers=headers)
        elif history_type == "Tumor":
            response = requests.get(
                f"{API_URL}/history/tumor", headers=headers)
        else:
            response = requests.get(
                f"{API_URL}/history/treatment", headers=headers)

        if response.status_code == 200:
            data = response.json()

            if len(data) > 0:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                st.metric("Total Records", len(df))
            else:
                st.warning("No records found.")
        else:
            st.error(response.text)
