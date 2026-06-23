# Heart-Disease

심장질환 위험 예측 머신러닝 프로젝트. UCI Heart Failure Prediction 데이터셋을 전처리 후 여러 분류 모델을 비교하고, 최종 모델을 Streamlit 앱으로 서빙한다.

## 데이터

- `dataset/heart.csv`: 원본 데이터 (Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease)
- `dataset/heart_train*.csv`, `heart_test*.csv`: 학습/평가 분할 및 스케일러(Standard/MinMax/Robust)별 전처리 데이터

## 작업 흐름

1. **전처리** (`heart_preprocessing.ipynb`): 결측치/이상치 처리, 범주형 인코딩, 스케일링
2. **모델링** (`heart_modeling.ipynb`): Logistic Regression, Random Forest, GradientBoosting, XGBoost, LightGBM 학습 및 GridSearchCV 튜닝, 성능 비교(Accuracy/ROC-AUC), 최종 모델(GradientBoosting) 선정 및 `models/`에 저장
3. **서빙** (`streamlit_heart.py`): 환자 정보를 입력받아 심장질환 위험을 예측하는 웹 앱

## Streamlit 앱

```bash
pip install -r requirements.txt
streamlit run streamlit_heart.py
```

3개 탭으로 구성:
- **예측**: 환자 정보 입력 후 GradientBoosting 모델로 심장질환 위험 예측
- **EDA**: 변수 간 상관관계, 타겟별 분포, 이상치, ydata-profiling 리포트
- **모델 비교**: 모델별 Accuracy/ROC-AUC 비교표, feature importance, confusion matrix 등

## 디렉터리 구조

```
dataset/    원본 및 전처리된 데이터
models/     학습된 모델(gb_model.pkl)과 feature 목록
output/     EDA/모델링 결과 이미지 및 리포트 (output/heart-figures/)
```
