import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("catboost_disease_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
symptom_weights = joblib.load("symptom_weights.pkl")
all_symptoms = joblib.load("all_symptoms.pkl")

# Preprocess symptom input
def clean_symptom(sym):
    return sym.strip().replace("_", "").replace(" ", "").lower()

def encode_input(symptoms, all_symptoms, symptom_weights):
    features = {s: 0 for s in all_symptoms}
    for sym in symptoms:
        clean = clean_symptom(sym)
        if clean in features:
            features[clean] = symptom_weights.get(clean, 1)
    return pd.DataFrame([features])

# Title
st.title("🩺 Disease Prediction System (CatBoost)")

# Symptom selection
st.markdown("### Select up to 10 symptoms:")
selected_symptoms = st.multiselect(
    "Choose symptoms:",
    options=sorted(list(set(all_symptoms))),  # show symptoms in dropdown
    max_selections=10,
    help="You can select up to 10 symptoms."
)

# Predict button
if st.button("Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        input_df = encode_input(selected_symptoms, all_symptoms, symptom_weights)
        pred_encoded = model.predict(input_df)[0]
        pred_disease = label_encoder.inverse_transform([pred_encoded])[0]
        st.success(f"🩺 Predicted Disease: **{pred_disease}**")
