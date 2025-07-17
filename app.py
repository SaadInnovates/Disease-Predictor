import streamlit as st
import joblib
import pandas as pd

# 🔄 Load trained model and preprocessing assets
model = joblib.load("catboost_disease_model.pkl")
symptom_weights = joblib.load("symptom_weights.pkl")
all_symptoms = joblib.load("all_symptoms.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# 🧠 Preprocess symptoms
def clean_text(text):
    return text.strip().replace("_", "").replace(" ", "").lower()

def encode_input(symptoms):
    feature_dict = {sym: 0 for sym in all_symptoms}
    for sym in symptoms:
        clean_sym = clean_text(sym)
        if clean_sym in symptom_weights:
            feature_dict[clean_sym] = symptom_weights[clean_sym]
    return pd.DataFrame([feature_dict])

# 🚀 Streamlit UI
st.title("🩺 Disease Prediction from Symptoms")
st.markdown("Select symptoms below and click **Predict Disease**.")

# 👇 Multiselect dropdown
selected_symptoms = st.multiselect(
    "Choose symptoms:",
    options=sorted(symptom_weights.keys()),
    help="Start typing to filter symptoms..."
)

if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        x_input = encode_input(selected_symptoms)
        pred_encoded = model.predict(x_input)[0]
        pred_disease = label_encoder.inverse_transform([int(pred_encoded)])[0]
        st.success(f"🧾 Predicted Disease: **{pred_disease}**")
