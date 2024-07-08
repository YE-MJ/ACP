import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from catboost import CatBoostClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, make_scorer
from scipy.stats import randint

# preprocessing_feature_extraction 디렉토리 경로 설정
input_directory_path = "./feature/combination_ACP_feature/"

# MCC 값을 저장할 리스트
mcc_values_cat = []
# Accuracy 값을 저장할 리스트
accuracy_values_cat = []
# Specificity 값을 저장할 리스트
specificity_values_cat = []
# Sensitivity 값을 저장할 리스트
sensitivity_values_cat = []
# 최적의 파라미터와 해당 파라미터로 얻은 MCC 값을 저장할 리스트
best_params_list_cat = []
best_mcc_list_cat = []
best_accuracy_list_cat = []
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

    # CatBoost 모델 생성
    model_cat = CatBoostClassifier()

    # RandomizedSearchCV를 사용하여 파라미터 최적화
    param_distributions_cat = {
        'iterations': randint(100, 1000),
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]
    }
    
    # MCC를 평가 지표로 설정
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    random_search_cat = RandomizedSearchCV(model_cat, param_distributions_cat, n_iter=10, scoring=mcc_scorer, cv=5, random_state=42)
    random_search_cat.fit(X, y)

    # 최적의 모델을 추출하여 교차 검증 예측을 수행
    best_model_cat = random_search_cat.best_estimator_
    y_pred_cat = cross_val_predict(best_model_cat, X, y, cv=5)
    
    # MCC 값 계산
    mcc_cat = matthews_corrcoef(y, y_pred_cat)
    mcc_values_cat.append(mcc_cat)

    # Accuracy 값 계산
    accuracy_cat = accuracy_score(y, y_pred_cat)
    accuracy_values_cat.append(accuracy_cat)

    # Confusion matrix를 사용하여 Specificity와 Sensitivity 계산
    tn, fp, fn, tp = confusion_matrix(y, y_pred_cat).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    specificity_values_cat.append(specificity)
    sensitivity_values_cat.append(sensitivity)

    # 최적의 파라미터와 해당 파라미터로 얻은 MCC, Accuracy, Specificity, Sensitivity 값을 저장
    best_params_list_cat.append(random_search_cat.best_params_)
    best_mcc_list_cat.append(mcc_cat)
    best_accuracy_list_cat.append(accuracy_cat)

    print(f"File '{file_path}' MCC (CatBoost): {mcc_cat}, Accuracy (CatBoost): {accuracy_cat}, Best Params (CatBoost): {random_search_cat.best_params_}")

# 결과를 txt 파일에 저장
output_file_path = "./catboost_combi_ACP.txt"
with open(output_file_path, 'w') as f:
    for dataset, params, mcc, accuracy, spec, sens in zip(dataset_names, best_params_list_cat, best_mcc_list_cat, accuracy_values_cat, specificity_values_cat, sensitivity_values_cat):
        f.write(f"Dataset: {dataset}\nParams: {params}\nMCC: {mcc}\nAccuracy: {accuracy}\nSpecificity: {spec}\nSensitivity: {sens}\n\n")

print(f"MCC, Accuracy, Specificity, and Sensitivity values and best parameters saved to '{output_file_path}'")
