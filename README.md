# Heart-Disease

심장질환 위험 예측 머신러닝 프로젝트. UCI Heart Failure Prediction 데이터셋을 전처리 후 여러 분류 모델을 비교하고, 최종 모델을 Streamlit 앱으로 서빙한다.

## 데이터

- `dataset/heart.csv`: 원본 데이터 (Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease)
- `dataset/heart_train*.csv`, `heart_test*.csv`: 학습/평가 분할 및 스케일러(Standard/MinMax/Robust)별 전처리 데이터

## 작업 흐름

1. **전처리** (`heart_preprocessing.ipynb`): 결측치/이상치 처리, 범주형 인코딩, 스케일링
2. **모델링** (`heart_modeling.ipynb`): Logistic Regression, Random Forest, GradientBoosting, XGBoost, LightGBM 학습 및 GridSearchCV 튜닝, 성능 비교(Accuracy/ROC-AUC), 최종 모델(GradientBoosting) 선정 및 `models/`에 저장
3. **서빙** (`streamlit_heart.py`): 환자 정보를 입력받아 심장질환 위험을 예측하는 웹 앱

## 전처리 상세 (`heart_preprocessing.ipynb`)

1. **결측치 처리**
   - `isnull().sum()` 기준 표준 결측치(NaN)는 없음.
   - 단, `RestingBP`와 `Cholesterol`에 의학적으로 불가능한 `0` 값이 결측치 대용으로 들어있음을 확인.
   - `RestingBP == 0`인 행(1건)은 삭제, `Cholesterol == 0`은 `NaN`으로 변환 후 **타깃(HeartDisease)별 그룹 median**으로 대체(클래스별 분포 차이를 보존하기 위해 전체 median 대신 그룹별 median 사용).
2. **중복 행**: `duplicated()` 확인 결과 중복 없음.
3. **이상치 확인**: `Age, RestingBP, Cholesterol, MaxHR, Oldpeak`에 대해 boxplot + IQR(1.5×IQR) 기준으로 이상치 개수만 확인하고, 임상 데이터 특성상 실제 환자값일 가능성이 높아 별도 제거는 하지 않음.
4. **다중공선성 분석**: 범주형 변수를 임시로 LabelEncoding한 뒤 상관관계 히트맵을 그려 변수 간 다중공선성 여부를 점검.
5. **타깃 대비 분포 시각화**: 수치형 변수별로 `HeartDisease` 값에 따른 분포(histplot)를 비교해 변별력이 있는 변수를 확인.
6. **범주형 인코딩**
   - 이진 변수(`Sex`, `ExerciseAngina`)는 0/1 매핑.
   - 다범주 변수(`ChestPainType`, `RestingECG`, `ST_Slope`)는 `pd.get_dummies(drop_first=True)`로 원-핫 인코딩.
7. **데이터 프로파일링**: `ydata-profiling`으로 전체 변수 리포트 생성 (`output/heart-figures/profile_report.html`).
8. **Train/Test 분리**: `train_test_split(test_size=0.2, random_state=42)` → train 733건 / test 184건.
9. **스케일링**: `StandardScaler`, `MinMaxScaler`, `RobustScaler` 3종을 **train 기준으로만 fit**하고 train/test에 transform하여 데이터 누수를 방지.
10. **저장 결과**
    - 트리 계열 모델용: 인코딩만 적용한 원본 그대로 (`heart_train.csv`, `heart_test.csv`)
    - 스케일 민감 모델용: 스케일러 3종 각각 적용한 버전 (`heart_train_scaled/minmax/robust.csv` 등)을 모두 저장해 모델링 단계에서 비교

## 모델링 및 성능 비교 (`heart_modeling.ipynb`)

### 1. Logistic Regression (해석용 베이스라인)
- 스케일링된 데이터(StandardScaler)로 학습, 회귀 계수로 변수별 영향 방향/크기 확인.
- Accuracy 0.88 / ROC-AUC 0.920

**스케일러 비교** (Logistic Regression 기준 Accuracy / ROC-AUC)

| Scaler | Accuracy | ROC-AUC |
|---|---|---|
| MinMaxScaler | 0.8750 | 0.9218 |
| StandardScaler | 0.8804 | 0.9204 |
| RobustScaler | 0.8804 | 0.9204 |

→ 큰 차이는 없으나 최종 비교표에는 StandardScaler 기준 결과를 사용.

### 2. Random Forest (성능용)
- 인코딩만 한 원본(비스케일링) 데이터로 학습 후 `GridSearchCV(cv=5, scoring='roc_auc')`로 튜닝.
- 튜닝 그리드: `n_estimators=[100,200,400]`, `max_depth=[4,8,12,None]`, `max_features=['sqrt','log2']`
- Best params: `max_depth=None, max_features='sqrt', n_estimators=100`
- Accuracy 0.88 / ROC-AUC 0.929 (튜닝 전후 동일 — 기본값이 최적과 일치)
- Feature importance 시각화 포함

### 3. Boosting 계열 (GradientBoosting / XGBoost / LightGBM)
GridSearchCV로 모델별 하이퍼파라미터 튜닝:

| 모델 | Best Params | Accuracy | ROC-AUC |
|---|---|---|---|
| GradientBoosting | `learning_rate=0.1, max_depth=2, n_estimators=100` | 0.8967 | 0.9456 |
| XGBoost | `learning_rate=0.05, max_depth=3, n_estimators=200, subsample=1.0` | 0.8859 | 0.9466 |
| LightGBM | `learning_rate=0.05, n_estimators=100, num_leaves=15` | 0.8750 | 0.9470 |

### 4. 전체 모델 비교

| 모델 | Accuracy | ROC-AUC |
|---|---|---|
| LightGBM | 0.8750 | 0.9470 |
| XGBoost | 0.8859 | 0.9466 |
| **GradientBoosting** | **0.8967** | **0.9456** |
| Random Forest (tuned) | 0.8804 | 0.9294 |
| Logistic Regression | 0.8804 | 0.9204 |

- ROC-AUC만 보면 LightGBM/XGBoost가 근소하게 높지만, **Accuracy 및 confusion matrix 기준 정밀도·재현율 균형**(특히 클래스 0/1 모두에서 안정적인 precision/recall)이 가장 좋은 **GradientBoosting**을 최종 모델로 채택.
- GradientBoosting test set 성능: precision 0.87(클래스 0)/0.91(클래스 1), recall 0.86(클래스 0)/0.92(클래스 1), accuracy 0.90.

### 5. 최종 모델(GradientBoosting) 상세 분석
- Feature importance, 테스트셋 샘플 40건에 대한 실제값 vs 예측확률 시각화로 모델 신뢰도 확인.
- 임의 환자 2명(고위험/저위험 케이스)에 대한 예측 검증 → 고위험 케이스 예측확률 0.94, 저위험 케이스 0.05로 직관과 일치.
- 최종 모델과 feature 목록을 `models/gb_model.pkl`, `models/feature_columns.pkl`로 저장.

## Streamlit 앱

배포된 앱: https://heartdiseasemachine.streamlit.app/

로컬 실행:
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
