import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, make_scorer, confusion_matrix
from scipy.stats import randint

# preprocessing_feature_extraction 디렉토리 경로 설정
input_directory_path = "./feature/embedding/train/"

# MCC 값을 저장할 리스트
mcc_values_ada = []
# Accuracy 값을 저장할 리스트
accuracy_values_ada = []
# Specificity 값을 저장할 리스트
specificity_values_ada = []
# Sensitivity 값을 저장할 리스트
sensitivity_values_ada = []
# 최적의 파라미터와 해당 파라미터로 얻은 MCC 값을 저장할 리스트
best_params_list_ada = []
best_mcc_list_ada = []
best_accuracy_list_ada = []
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

    # Adaboost 모델 생성
    model_ada = AdaBoostClassifier()

    # RandomizedSearchCV를 사용하여 파라미터 최적화 및 교차 검증 수행
    param_distributions_ada = {
        'n_estimators': randint(50, 500),
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'algorithm': ['SAMME', 'SAMME.R']
    }
    
    # MCC를 평가 지표로 설정
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    random_search_ada = RandomizedSearchCV(model_ada, param_distributions_ada, n_iter=10, scoring=mcc_scorer, cv=5, random_state=42)
    
    # RandomizedSearchCV를 사용하여 최적의 모델을 찾음
    random_search_ada.fit(X, y)
    
    # 최적의 모델을 추출하여 교차 검증 예측을 수행
    best_model_ada = random_search_ada.best_estimator_
    y_pred_ada = cross_val_predict(best_model_ada, X, y, cv=5)
    
    # MCC 값 계산
    mcc_ada = matthews_corrcoef(y, y_pred_ada)
    mcc_values_ada.append(mcc_ada)

    # Accuracy 값 계산
    accuracy_ada = accuracy_score(y, y_pred_ada)
    accuracy_values_ada.append(accuracy_ada)

    # Confusion matrix를 사용하여 Specificity와 Sensitivity 계산
    tn, fp, fn, tp = confusion_matrix(y, y_pred_ada).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    specificity_values_ada.append(specificity)
    sensitivity_values_ada.append(sensitivity)

    # 최적의 파라미터와 해당 파라미터로 얻은 MCC, Accuracy, Specificity, Sensitivity 값을 저장
    best_params_list_ada.append(random_search_ada.best_params_)
    best_mcc_list_ada.append(mcc_ada)
    best_accuracy_list_ada.append(accuracy_ada)

    print(f"File '{file_path}' MCC (Adaboost): {mcc_ada}, Accuracy (Adaboost): {accuracy_ada}, Best Params (Adaboost): {random_search_ada.best_params_}")

# 결과를 txt 파일에 저장
output_file_path = "./adaboost_combi_ACP.txt"
with open(output_file_path, 'w') as f:
    for dataset, params, mcc, accuracy, spec, sens in zip(dataset_names, best_params_list_ada, best_mcc_list_ada, accuracy_values_ada, specificity_values_ada, sensitivity_values_ada):
        f.write(f"Dataset: {dataset}\nParams: {params}\nMCC: {mcc}\nAccuracy: {accuracy}\nSpecificity: {spec}\nSensitivity: {sens}\n\n")

print(f"MCC, Accuracy, Specificity, and Sensitivity values and best parameters saved to '{output_file_path}'")
