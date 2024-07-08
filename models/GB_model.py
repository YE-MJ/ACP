import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from scipy.stats import randint

# preprocessing_feature_extraction 디렉토리 경로 설정
input_directory_path = "./feature/embedding/train/"

# MCC 값을 저장할 리스트
mcc_values_gb = []
# Accuracy 값을 저장할 리스트
accuracy_values_gb = []
# Specificity 값을 저장할 리스트
specificity_values_gb = []
# Sensitivity 값을 저장할 리스트
sensitivity_values_gb = []
# 최적의 파라미터와 해당 파라미터로 얻은 MCC 값을 저장할 리스트
best_params_list_gb = []
best_mcc_list_gb = []
best_accuracy_list_gb = []
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

    # Gradient Boosting 모델 생성
    model_gb = GradientBoostingClassifier()

    # RandomizedSearchCV를 사용하여 파라미터 최적화
    param_distributions_gb = {
        'n_estimators': randint(50, 500),
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': randint(3, 10),
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    random_search_gb = RandomizedSearchCV(model_gb, param_distributions_gb, n_iter=10, scoring='accuracy', cv=5, random_state=42)
    random_search_gb.fit(X, y)

    # 최적의 모델을 추출하여 교차 검증 예측을 수행
    best_model_gb = random_search_gb.best_estimator_
    y_pred_gb = cross_val_predict(best_model_gb, X, y, cv=5)
    
    # MCC 값 계산
    mcc_gb = matthews_corrcoef(y, y_pred_gb)
    mcc_values_gb.append(mcc_gb)

    # Accuracy 값 계산
    accuracy_gb = accuracy_score(y, y_pred_gb)
    accuracy_values_gb.append(accuracy_gb)

    # Confusion matrix를 사용하여 Specificity와 Sensitivity 계산
    tn, fp, fn, tp = confusion_matrix(y, y_pred_gb).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    specificity_values_gb.append(specificity)
    sensitivity_values_gb.append(sensitivity)

    # 최적의 파라미터와 해당 파라미터로 얻은 MCC, Accuracy, Specificity, Sensitivity 값을 저장
    best_params_list_gb.append(random_search_gb.best_params_)
    best_mcc_list_gb.append(mcc_gb)
    best_accuracy_list_gb.append(accuracy_gb)

    print(f"File '{file_path}' MCC (GB): {mcc_gb}, Accuracy (GB): {accuracy_gb}, Specificity (GB): {specificity}, Sensitivity (GB): {sensitivity}")

# 결과를 txt 파일에 저장
output_file_path = "./gb_combi_ACP.txt"
with open(output_file_path, 'w') as f:
    for dataset, params_gb, mcc_gb, accuracy_gb, spec_gb, sens_gb in zip(dataset_names, best_params_list_gb, best_mcc_list_gb, accuracy_values_gb, specificity_values_gb, sensitivity_values_gb):
        f.write(f"Dataset: {dataset}\nParams: {params_gb}\nMCC: {mcc_gb}\nAccuracy: {accuracy_gb}\nSpecificity: {spec_gb}\nSensitivity: {sens_gb}\n\n")

print(f"MCC, Accuracy, Specificity, and Sensitivity values and best parameters for GB saved to '{output_file_path}'")
