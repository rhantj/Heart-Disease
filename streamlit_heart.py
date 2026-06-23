import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

MODEL_PATH = "models/gb_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"
FIGURES_DIR = "output/heart-figures"

MODEL_COMPARISON = pd.DataFrame({
    "Accuracy": [0.875000, 0.885870, 0.896739, 0.880435, 0.880435],
    "ROC-AUC": [0.947049, 0.946553, 0.945561, 0.929377, 0.920387],
}, index=["LightGBM", "XGBoost", "GradientBoosting", "Random Forest", "Logistic Regression"])


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


st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")
st.title("Heart Disease Prediction")

tab_predict, tab_eda, tab_compare = st.tabs(["예측", "EDA", "모델 비교"])

with tab_eda:
    st.subheader("데이터 탐색")
    col1, col2 = st.columns(2)
    with col1:
        st.image(f"{FIGURES_DIR}/correlation_heatmap.png", caption="변수 간 상관관계")
    with col2:
        st.image(f"{FIGURES_DIR}/distribution_by_target.png", caption="HeartDisease 그룹별 변수 분포")

    st.image(f"{FIGURES_DIR}/outliers_boxplot.png", caption="이상치 분포 (Boxplot)", use_container_width=True)


with tab_compare:
    st.subheader("모델별 성능 비교")
    st.dataframe(
        MODEL_COMPARISON.sort_values("ROC-AUC", ascending=False).style.format("{:.4f}"),
        use_container_width=True,
    )

    st.subheader("최종 모델(GradientBoosting) 분석")
    col1, col2 = st.columns(2)
    with col1:
        st.image(f"{FIGURES_DIR}/gb_feature_importance.png", caption="GradientBoosting Feature Importance")
    with col2:
        st.image(f"{FIGURES_DIR}/confusion_matrices.png", caption="Confusion Matrices (모델별)")

    col3, col4 = st.columns(2)
    with col3:
        st.image(f"{FIGURES_DIR}/logreg_coefficients.png", caption="Logistic Regression Coefficients")
    with col4:
        st.image(f"{FIGURES_DIR}/rf_feature_importance.png", caption="Random Forest Feature Importance")

    st.image(f"{FIGURES_DIR}/gb_sample_prediction.png", caption="GradientBoosting 샘플 예측 결과", use_container_width=True)

with tab_predict:
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
