import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/gb_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, feature_columns


def encode_input(raw_input: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    encoded = raw_input.copy()
    encoded["Sex"] = encoded["Sex"].map({"M": 1, "F": 0})
    encoded["ExerciseAngina"] = encoded["ExerciseAngina"].map({"Y": 1, "N": 0})
    encoded = pd.get_dummies(
        encoded, columns=["ChestPainType", "RestingECG", "ST_Slope"],
        drop_first=True, dtype=int,
    )
    return encoded.reindex(columns=feature_columns, fill_value=0)


st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("Heart Disease Prediction")
st.caption("GradientBoosting 모델을 이용한 심장질환 위험 예측")

model, feature_columns = load_model()

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain_type = st.selectbox("ChestPainType", ["ATA", "NAP", "ASY", "TA"])
        resting_bp = st.number_input("RestingBP", min_value=0, max_value=250, value=120)
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=700, value=200)

    with col2:
        fasting_bs = st.selectbox("FastingBS (>120 mg/dl)", [0, 1])
        resting_ecg = st.selectbox("RestingECG", ["Normal", "ST", "LVH"])
        max_hr = st.number_input("MaxHR", min_value=60, max_value=220, value=150)
        exercise_angina = st.selectbox("ExerciseAngina", ["N", "Y"])
        oldpeak = st.number_input("Oldpeak", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
        st_slope = st.selectbox("ST_Slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("예측하기")

if submitted:
    raw_input = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain_type,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }])

    encoded_input = encode_input(raw_input, feature_columns)
    prediction = model.predict(encoded_input)[0]
    probability = model.predict_proba(encoded_input)[0, 1]

    st.divider()
    if prediction == 1:
        st.error(f"심장질환 위험이 있습니다. (예측 확률: {probability:.1%})")
    else:
        st.success(f"심장질환 위험이 낮습니다. (예측 확률: {probability:.1%})")
