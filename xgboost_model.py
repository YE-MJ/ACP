import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from scipy.stats import randint
import numpy as np

# 데이터셋 디렉토리 경로 설정
input_directory_path = "./feature/combination_ACP_feature/"

# MCC 값을 저장할 리스트
mcc_values_xgb = []
# Accuracy 값을 저장할 리스트
accuracy_values_xgb = []
# Specificity 값을 저장할 리스트
specificity_values_xgb = []
# Sensitivity 값을 저장할 리스트
sensitivity_values_xgb = []
# 최적의 파라미터와 해당 파라미터로 얻은 MCC 값을 저장할 리스트
best_params_list_xgb = []
best_mcc_list_xgb = []
best_accuracy_list_xgb = []
# 각 데이터셋 파일명 저장할 리스트
dataset_names = []

# 디렉토리 내의 모든 CSV 파일 경로 가져오기
csv_files = [f for f in os.listdir(input_directory_path) if f.endswith(".csv")]

# 각 CSV 파일에 대해 작업 수행
for file_name in csv_files:
    file_path = os.path.join(input_directory_path, file_name)
    dataset_names.append(file_name)  # 파일명 리스트에 추가

    # CSV 파일 불러오기
    df = pd.read_csv(file_path)

    # 특성과 타겟 데이터 분리
    X = df.drop(['name', 'target'], axis=1)
    y = df['target']

    model_xgb = XGBClassifier()

    # RandomizedSearchCV를 사용하여 파라미터 최적화
    param_distributions_xgb = {
        'n_estimators': randint(100, 1000),
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'reg_alpha': [0, 0.1, 0.5, 1, 2],
        'reg_lambda': [0, 0.1, 0.5, 1, 2]
    }
    random_search_xgb = RandomizedSearchCV(model_xgb, param_distributions_xgb, n_iter=10, scoring='accuracy', cv=5, random_state=42)
    random_search_xgb.fit(X, y)

    # 최적의 모델로 예측 확률 계산
    best_model_xgb = random_search_xgb.best_estimator_
    y_pred_xgb = cross_val_predict(best_model_xgb, X, y, cv=5)

    # Confusion Matrix 계산
    cm = confusion_matrix(y, y_pred_xgb)
    tn, fp, fn, tp = cm.ravel()

    # MCC 값 계산
    mcc_xgb = matthews_corrcoef(y, y_pred_xgb)
    mcc_values_xgb.append(mcc_xgb)

    # Accuracy 값 계산
    accuracy_xgb = accuracy_score(y, y_pred_xgb)
    accuracy_values_xgb.append(accuracy_xgb)

    # Specificity 계산
    specificity = tn / (tn + fp)
    specificity_values_xgb.append(specificity)

    # Sensitivity 계산
    sensitivity = tp / (tp + fn)
    sensitivity_values_xgb.append(sensitivity)

    # 최적의 파라미터와 해당 파라미터로 얻은 MCC 및 Accuracy 값을 저장
    best_params_list_xgb.append(random_search_xgb.best_params_)

    print(f"파일 '{file_path}' MCC (XGBoost): {mcc_xgb}, Accuracy (XGBoost): {accuracy_xgb}, Specificity (Specificity): {specificity}, Sensitivity (Sensitivity): {sensitivity}")

# 결과를 txt 파일에 저장
output_file_path = "./xgboost_combi_ACP.txt"
with open(output_file_path, 'w') as f:
    for dataset, params_xgb, mcc_xgb, accuracy_xgb, specificity_xgb, sensitivity_xgb in zip(dataset_names, best_params_list_xgb, mcc_values_xgb, accuracy_values_xgb, specificity_values_xgb, sensitivity_values_xgb):
        f.write(f"Dataset: {dataset}\nParams: {params_xgb}\nMCC: {mcc_xgb}\nAccuracy: {accuracy_xgb}\nSpecificity: {specificity_xgb}\nSensitivity: {sensitivity_xgb}\n\n")

print(f"MCC, Accuracy, Specificity, Sensitivity 값 및 최적 파라미터를 '{output_file_path}'에 저장하였습니다.")
