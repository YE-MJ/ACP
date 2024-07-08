import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from scipy.stats import uniform
import numpy as np

# 데이터셋 디렉토리 경로 설정
input_directory_path = "./feature/csv/train/"

# MCC 값을 저장할 리스트
mcc_values_svc = []
# Accuracy 값을 저장할 리스트
accuracy_values_svc = []
# Specificity 값을 저장할 리스트
specificity_values_svc = []
# Sensitivity 값을 저장할 리스트
sensitivity_values_svc = []
# 최적의 파라미터와 해당 파라미터로 얻은 MCC 값을 저장할 리스트
best_params_list_svc = []
best_mcc_list_svc = []
best_accuracy_list_svc = []
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

    model_svc = SVC()

    # RandomizedSearchCV를 사용하여 파라미터 최적화
    param_distributions_svc = {
        'C': uniform(0.1, 10),
        'gamma': uniform(0.01, 1),
        'kernel': ['linear', 'rbf']
    }
    random_search_svc = RandomizedSearchCV(model_svc, param_distributions_svc, n_iter=10, scoring='accuracy', cv=5, random_state=42)
    random_search_svc.fit(X, y)

    # 최적의 모델로 예측 확률 계산
    best_model_svc = random_search_svc.best_estimator_
    y_pred_svc = cross_val_predict(best_model_svc, X, y, cv=5)

    # Confusion Matrix 계산
    cm = confusion_matrix(y, y_pred_svc)
    tn, fp, fn, tp = cm.ravel()

    # MCC 값 계산
    mcc_svc = matthews_corrcoef(y, y_pred_svc)
    mcc_values_svc.append(mcc_svc)

    # Accuracy 값 계산
    accuracy_svc = accuracy_score(y, y_pred_svc)
    accuracy_values_svc.append(accuracy_svc)

    # Specificity 계산
    specificity = tn / (tn + fp)
    specificity_values_svc.append(specificity)

    # Sensitivity 계산
    sensitivity = tp / (tp + fn)
    sensitivity_values_svc.append(sensitivity)

    # 최적의 파라미터와 해당 파라미터로 얻은 MCC 및 정확도 값을 저장
    best_params_list_svc.append(random_search_svc.best_params_)

    print(f"파일 '{file_path}' MCC (SVC): {mcc_svc}, Accuracy (SVC): {accuracy_svc}, Specificity (Specificity): {specificity}, Sensitivity (Sensitivity): {sensitivity}")

# 결과를 txt 파일에 저장
output_file_path = "./svc_feature_ACP.txt"
with open(output_file_path, 'w') as f:
    for dataset, params_svc, mcc_svc, accuracy_svc, specificity_svc, sensitivity_svc in zip(dataset_names, best_params_list_svc, mcc_values_svc, accuracy_values_svc, specificity_values_svc, sensitivity_values_svc):
        f.write(f"Dataset: {dataset}\nParams: {params_svc}\nMCC: {mcc_svc}\nAccuracy: {accuracy_svc}\nSpecificity: {specificity_svc}\nSensitivity: {sensitivity_svc}\n\n")

print(f"MCC, Accuracy, Specificity, Sensitivity 값 및 최적 파라미터를 '{output_file_path}'에 저장하였습니다.")
