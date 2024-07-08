import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from scipy.stats import randint

# 데이터셋 디렉토리 경로 설정
input_directory_path = "./feature/embedding/train/"

# MCC 값을 저장할 리스트
mcc_values_rf = []
# Accuracy 값을 저장할 리스트
accuracy_values_rf = []
# Specificity 값을 저장할 리스트
specificity_values_rf = []
# Sensitivity 값을 저장할 리스트
sensitivity_values_rf = []
# 최적의 파라미터와 해당 파라미터로 얻은 MCC 값을 저장할 리스트
best_params_list_rf = []
best_mcc_list_rf = []
best_accuracy_list_rf = []
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

    # RandomForest 모델 생성
    model_rf = RandomForestClassifier()

    # RandomizedSearchCV를 사용하여 파라미터 최적화
    param_distributions_rf = {
        'n_estimators': randint(100, 1000),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2']
    }
    random_search_rf = RandomizedSearchCV(model_rf, param_distributions_rf, n_iter=10, scoring='accuracy', cv=5, random_state=42)
    random_search_rf.fit(X,y)

    # 최적의 모델로 예측 확률 계산
    best_model_rf = random_search_rf.best_estimator_
    y_pred_rf = cross_val_predict(best_model_rf, X, y, cv=5)

    # Confusion Matrix 계산
    cm = confusion_matrix(y, y_pred_rf)
    tn, fp, fn, tp = cm.ravel()

    # MCC 값 계산
    mcc_rf = matthews_corrcoef(y, y_pred_rf)
    mcc_values_rf.append(mcc_rf)

    # 정확도 값 계산
    accuracy_rf = accuracy_score(y, y_pred_rf)
    accuracy_values_rf.append(accuracy_rf)

    # Specificity 계산
    specificity = tn / (tn + fp)
    specificity_values_rf.append(specificity)

    # Sensitivity 계산
    sensitivity = tp / (tp + fn)
    sensitivity_values_rf.append(sensitivity)

    best_params_list_rf.append(random_search_rf.best_params_)
    best_mcc_list_rf.append(mcc_rf)
    best_accuracy_list_rf.append(accuracy_rf)

    print(f"파일 '{file_path}' MCC (RandomForest): {mcc_rf}, 정확도 (RandomForest): {accuracy_rf}, 특이도 (Specificity): {specificity}, 민감도 (Sensitivity): {sensitivity}")

# 결과를 txt 파일에 저장
output_file_path = "./RF_combi_ACP.txt"
with open(output_file_path, 'w') as f:
    for dataset, params_rf, mcc_rf, accuracy_rf, specificity_rf, sensitivity_rf in zip(dataset_names, best_params_list_rf, best_mcc_list_rf, accuracy_values_rf, specificity_values_rf, sensitivity_values_rf):
        f.write(f"Dataset: {dataset}\nParams: {params_rf}\nMCC: {mcc_rf}\nAccuracy: {accuracy_rf}\nSpecificity: {specificity_rf}\nSensitivity: {sensitivity_rf}\n\n")
        
print(f"MCC, Accuracy, Specificity, and Sensitivity values and best parameters saved to '{output_file_path}'")