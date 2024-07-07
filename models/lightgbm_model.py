import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from scipy.stats import randint

# 데이터셋 디렉토리 경로 설정
input_directory_path = "./feature/embedding/train/"

# MCC 값을 저장할 리스트
mcc_values_lgbm = []
# Accuracy 값을 저장할 리스트
accuracy_values_lgbm = []
# Specificity 값을 저장할 리스트
specificity_values_lgbm = []
# Sensitivity 값을 저장할 리스트
sensitivity_values_lgbm = []
# 최적의 파라미터와 해당 파라미터로 얻은 MCC 값을 저장할 리스트
best_params_list_lgbm = []
best_mcc_list_lgbm = []
best_accuracy_list_lgbm = []
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

    # LightGBM 모델 생성
    model_lgbm = LGBMClassifier()

    # RandomizedSearchCV를 사용하여 파라미터 최적화
    param_distributions_lgbm = {
        'n_estimators': randint(100, 1000),
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1, 2],
        'reg_lambda': [0, 0.1, 0.5, 1, 2]
    }
    random_search_lgbm = RandomizedSearchCV(model_lgbm, param_distributions_lgbm, n_iter=10, scoring='accuracy', cv=5, random_state=42)
    random_search_lgbm.fit(X, y)
    
    # 최적의 모델로 예측 확률 계산
    best_model_lgbm = random_search_lgbm.best_estimator_
    y_pred_lgbm = cross_val_predict(best_model_lgbm, X, y, cv=5)
    
    # MCC 값 계산
    mcc_lgbm = matthews_corrcoef(y, y_pred_lgbm)
    mcc_values_lgbm.append(mcc_lgbm)

    # 정확도 값 계산
    accuracy_lgbm = accuracy_score(y, y_pred_lgbm)
    accuracy_values_lgbm.append(accuracy_lgbm)

    # Confusion Matrix를 사용하여 Specificity와 Sensitivity 계산
    tn, fp, fn, tp = confusion_matrix(y, y_pred_lgbm).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    specificity_values_lgbm.append(specificity)
    sensitivity_values_lgbm.append(sensitivity)

    # 최적의 파라미터와 해당 파라미터로 얻은 MCC, 정확도, Specificity, Sensitivity 값을 저장
    best_params_list_lgbm.append(random_search_lgbm.best_params_)
    best_mcc_list_lgbm.append(mcc_lgbm)
    best_accuracy_list_lgbm.append(accuracy_lgbm)

    print(f"파일 '{file_path}' MCC (LightGBM): {mcc_lgbm}, 정확도 (LightGBM): {accuracy_lgbm}, 특이도 (LightGBM): {specificity}, 민감도 (LightGBM): {sensitivity}")

# 결과를 txt 파일에 저장
output_file_path = "./lightgbm_combi_ACP.txt"
with open(output_file_path, 'w', encoding='utf-8') as f:
    for dataset, params, mcc, accuracy, spec, sens in zip(dataset_names, best_params_list_lgbm, best_mcc_list_lgbm, accuracy_values_lgbm, specificity_values_lgbm, sensitivity_values_lgbm):
        f.write(f"Dataset: {dataset}\nParams: {params}\nMCC: {mcc}\nAccuracy: {accuracy}\nSpecificity: {spec}\nSensitivity: {sens}\n\n")

print(f"MCC, Accuracy, Specificity, and Sensitivity values and best parameters saved to '{output_file_path}'")
