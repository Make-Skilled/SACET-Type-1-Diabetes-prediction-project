import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# App Title
st.title("🩸 Blood Glucose Prediction App")

st.markdown("""
Predict blood glucose levels based on patient health attributes.
Upload a CSV file and enter values to get predictions.
""")

# File Upload
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(data.head())

    # Drop non-relevant or ID-like columns
    if 'Patient_ID' in data.columns:
        data = data.drop(columns=['Patient_ID'])

    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['Gender', 'Meal_Timing', 'Activity_Level', 'Diabetes_Type', 'Blood_Glucose_Status']
    
    for col in categorical_columns:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le  # Store label encoders for future decoding

    # Separate features and target variable
    X = data.drop(columns=['Blood_Glucose_Status'])
    y = data['Blood_Glucose_Status']

    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Train a Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Save Model & Scaler
    with open("rf_model.pkl", "wb") as model_file:
        pickle.dump(rf_classifier, model_file)
    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    with open("label_encoders.pkl", "wb") as encoder_file:
        pickle.dump(label_encoders, encoder_file)

    # Evaluate the model
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.subheader("Model Accuracy")
    st.write(f"📊 Accuracy: **{accuracy:.2f}**")

    # Prediction Section
    st.subheader("🎯 Predict Blood Glucose Status")

    user_input = {}

    # Categorical Inputs with Dropdown
    category_mappings = {
        "Gender": ["Female", "Male"],
        "Meal_Timing": ["Breakfast", "Lunch", "Dinner", "Snack"],
        "Activity_Level": ["Low", "Moderate", "High"],
        "Diabetes_Type": ["Type 1"]
    }

    for col in X.columns:
        if col in category_mappings:
            user_input[col] = st.selectbox(f"Select {col}:", category_mappings[col])
        else:
            user_input[col] = st.number_input(f"Enter {col}:", value=0.0, step=0.01)

    # Load saved model and scaler
    with open("rf_model.pkl", "rb") as model_file:
        rf_classifier = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with open("label_encoders.pkl", "rb") as encoder_file:
        label_encoders = pickle.load(encoder_file)

    if st.button("🔮 Predict"):
        # Convert categorical inputs to encoded values
        for col in category_mappings.keys():
            if col in user_input:
                user_input[col] = category_mappings[col].index(user_input[col])  # Encode category

        # Convert input to DataFrame
        input_df = pd.DataFrame([user_input])

        # Scale numerical features
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)

        # Make Prediction
        prediction = rf_classifier.predict(input_scaled_df)
        predicted_label = label_encoders['Blood_Glucose_Status'].inverse_transform(prediction)

        st.success(f"🩸 Predicted Blood Glucose Status: **{predicted_label[0]}**")

        # Suggest Precautions
        if predicted_label[0] == "High":
            st.warning("⚠ **Precaution 1:** Monitor blood glucose closely and take prescribed medication.")
            st.warning("⚠ **Precaution 2:** Follow a balanced diet low in refined sugars and high in fiber.")
            st.warning("⚠ **Precaution 3:** Engage in regular physical activity to improve insulin sensitivity.")
            st.warning("⚠ **Precaution 4:** Stay hydrated and avoid alcohol or sugary beverages.")
            st.warning("⚠ **Precaution 5:** Consult a healthcare provider for personalized diabetes management.")

        elif predicted_label[0] == "Low":
            st.warning("⚠ **Precaution 1:** Eat small, frequent meals to maintain stable glucose levels.")
            st.warning("⚠ **Precaution 2:** Carry a source of fast-acting sugar (glucose tablets, juice) for emergencies.")
            st.warning("⚠ **Precaution 3:** Avoid excessive alcohol intake, as it can cause blood sugar drops.")
            st.warning("⚠ **Precaution 4:** Monitor blood sugar levels regularly, especially before driving or exercising.")
            st.warning("⚠ **Precaution 5:** Adjust medication doses under a doctor's supervision if frequent lows occur.")

        else:
            st.info("✅ **Your glucose level is normal! Maintain a healthy lifestyle.**")
            st.info("✅ **Precaution 1:** Continue eating a balanced diet with whole grains, lean proteins, and healthy fats.")
            st.info("✅ **Precaution 2:** Stay physically active and manage stress effectively.")
            st.info("✅ **Precaution 3:** Keep a regular sleep schedule, as sleep affects glucose metabolism.")
            st.info("✅ **Precaution 4:** Stay hydrated and monitor any unusual symptoms.")
            st.info("✅ **Precaution 5:** Regularly visit your healthcare provider for routine checkups.")

