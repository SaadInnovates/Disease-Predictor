# 🩺 Disease Prediction Web App

A machine learning-powered disease diagnosis tool built with **CatBoostClassifier** and a **Streamlit** interface. Users can select symptoms, and the model predicts the most probable disease.

---

## 🚀 Features

- 🔍 Predicts diseases based on symptoms
- 💡 Uses a trained **CatBoostClassifier** model
- 🧠 Weighted symptom encoding for higher accuracy
- 💻 User-friendly Streamlit frontend
- 🛠️ Easily deployable and customizable

---

## 🧾 Dataset

- Dataset: `dataset.csv`
- Symptom severity weights: `Symptom-severity.csv`

The dataset contains disease-symptom mappings with cleaned and normalized symptom names. Severity scores are used for feature encoding.

---

## 🧠 Model Info

- Model: `CatBoostClassifier`
- Tuned hyperparameters:
  - `iterations=300`
  - `learning_rate=0.01`
  - `depth=4`
  - `l2_leaf_reg=6`
  - `bagging_temperature=0.8`
  - `random_strength=8`
- Accuracy: ✅ ~99.5% on test set
- Artifacts:
  - `catboost_custom_model.pkl`
  - `label_encoder.pkl`
  - `all_symptoms.pkl`
  - `symptom_weights.pkl`

---

## 🖥️ Installation

```bash
git clone https://github.com/your-username/disease-prediction-app.git
cd disease-prediction-app
pip install -r requirements.txt
