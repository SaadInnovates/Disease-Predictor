import streamlit as st
import pandas as pd
import joblib

# ✅ Load artifacts
model = joblib.load("catboost_disease_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
all_symptoms = joblib.load("all_symptoms.pkl")
symptom_weights = joblib.load("symptom_weights.pkl")

# ✅ Load CSVs
desc_df = pd.read_csv("symptom_Description.csv")
precaution_df = pd.read_csv("symptom_precaution.csv")

# ✅ Helper functions
def clean_symptom(symptom):
    return symptom.strip().replace("_", "").replace(" ", "").lower()

def encode_input(selected_symptoms, all_symptoms, symptom_weights):
    feature_dict = {sym: 0 for sym in all_symptoms}
    for sym in selected_symptoms:
        cleaned = clean_symptom(sym)
        if cleaned in symptom_weights:
            feature_dict[cleaned] = symptom_weights[cleaned]
    return pd.DataFrame([feature_dict])

def get_description(disease_name):
    match = desc_df[desc_df["Disease"].str.lower() == disease_name.lower()]
    return match["Description"].values[0] if not match.empty else "No description available."

def get_precautions(disease_name):
    match = precaution_df[precaution_df["Disease"].str.lower() == disease_name.lower()]
    if not match.empty:
        return [
            match["Precaution_1"].values[0],
            match["Precaution_2"].values[0],
            match["Precaution_3"].values[0],
            match["Precaution_4"].values[0]
        ]
    else:
        return ["Not available"] * 4

# ✅ Symptoms list (same as your full list)
raw_symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
    'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
    'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss',
    'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
    'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
    'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
    'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
    'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes',
    'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain',
    'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
    'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
    'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
    'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_ofurine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
    'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation',
    'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
    'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations',
    'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting',
    'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze', 'prognosis'
]
# ✅ Streamlit UI
st.title("🩺 Disease Prediction System")
selected_symptoms = st.multiselect(
    "Select your symptoms (min 5, max 17)", 
    raw_symptoms,
    help="Hold Ctrl (or ⌘ on Mac) to select multiple symptoms."
)

if st.button("Predict Disease"):
    num_symptoms = len(selected_symptoms)

    if num_symptoms < 3:
        st.warning("⚠️ Please select **at least 3 symptoms**.")
    elif num_symptoms > 17:
        st.warning("⚠️ Please select **no more than 17 symptoms**.")
    else:
        x_input = encode_input(selected_symptoms, all_symptoms, symptom_weights)
        probs = model.predict_proba(x_input)[0]

        # Get top 5 predictions
        top_indices = probs.argsort()[-5:][::-1]
        top_diseases = label_encoder.inverse_transform(top_indices)

        st.subheader("🔍 Based on your symptoms, you may have one of the following diseases:")

        for i in range(5):
            disease = top_diseases[i]
            probability = probs[top_indices[i]]
            description = get_description(disease)
            precautions = get_precautions(disease)

            st.markdown(f"""
            ### {i+1}. 🦠 **{disease}**
            - 🔢 Probability: **{probability:.2f}**
            - 📝 Description: {description}
            - 🛡️ **Precautions:**
                1. {precautions[0]}
                2. {precautions[1]}
                3. {precautions[2]}
                4. {precautions[3]}
            """)


        
        st.info("ℹ️ These are AI-based predictions and suggestions. Please consult a doctor for an official diagnosis.")

