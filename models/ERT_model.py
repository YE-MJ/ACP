import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from scipy.stats import randint

# preprocessing_feature_extraction 디렉토리 경로 설정
input_directory_path = "./feature/embedding/train/"

# MCC 값을 저장할 리스트
mcc_values_ert = []
# Accuracy 값을 저장할 리스트
accuracy_values_ert = []
# Specificity 값을 저장할 리스트
specificity_values_ert = []
# Sensitivity 값을 저장할 리스트
sensitivity_values_ert = []
# 최적의 파라미터와 해당 파라미터로 얻은 MCC 값을 저장할 리스트
best_params_list_ert = []
best_mcc_list_ert = []
best_accuracy_list_ert = []
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

    # ERT 모델 생성
    model_ert = ExtraTreesClassifier()

    # RandomizedSearchCV를 사용하여 파라미터 최적화
    param_distributions_ert = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(3, 10),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    random_search_ert = RandomizedSearchCV(model_ert, param_distributions_ert, n_iter=10, scoring='accuracy', cv=5, random_state=42)
    random_search_ert.fit(X, y)

    # 최적의 모델을 추출하여 교차 검증 예측을 수행
    best_model_ert = random_search_ert.best_estimator_
    y_pred_ert = cross_val_predict(best_model_ert, X, y, cv=5)
    
    # MCC 값 계산
    mcc_ert = matthews_corrcoef(y, y_pred_ert)
    mcc_values_ert.append(mcc_ert)

    # Accuracy 값 계산
    accuracy_ert = accuracy_score(y, y_pred_ert)
    accuracy_values_ert.append(accuracy_ert)

    # Confusion matrix를 사용하여 Specificity와 Sensitivity 계산
    tn, fp, fn, tp = confusion_matrix(y, y_pred_ert).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    specificity_values_ert.append(specificity)
    sensitivity_values_ert.append(sensitivity)

    # 최적의 파라미터와 해당 파라미터로 얻은 MCC, Accuracy, Specificity, Sensitivity 값을 저장
    best_params_list_ert.append(random_search_ert.best_params_)
    best_mcc_list_ert.append(mcc_ert)
    best_accuracy_list_ert.append(accuracy_ert)

    print(f"File '{file_path}' MCC (ERT): {mcc_ert}, Accuracy (ERT): {accuracy_ert}, Best Params (ERT): {random_search_ert.best_params_}")

# 결과를 txt 파일에 저장
output_file_path = "./ert_combi_ACP.txt"
with open(output_file_path, 'w') as f:
    for dataset, params_ert, mcc_ert, accuracy_ert, spec_ert, sens_ert in zip(dataset_names, best_params_list_ert, best_mcc_list_ert, accuracy_values_ert, specificity_values_ert, sensitivity_values_ert):
        f.write(f"Dataset: {dataset}\nParams: {params_ert}\nMCC: {mcc_ert}\nAccuracy: {accuracy_ert}\nSpecificity: {spec_ert}\nSensitivity: {sens_ert}\n\n")

print(f"MCC, Accuracy, Specificity, and Sensitivity values and best parameters for ERT saved to '{output_file_path}'")
