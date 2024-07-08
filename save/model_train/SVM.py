import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from scipy.stats import uniform
import joblib
import numpy as np

# 사용할 CSV 파일 경로 리스트 설정
csv_files = [
    './selected_feature/protein_embeddings_prott5.csv', './selected_feature/protein_embeddings_esm2.csv'
]

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
# 각 데이터셋 파일명 저장할 리스트
dataset_names = []

# 각 CSV 파일에 대해 작업 수행
for file_path in csv_files:
    try:
        file_name = os.path.basename(file_path)
        dataset_names.append(file_name)  # 파일명 리스트에 추가

        # CSV 파일 불러오기
        df = pd.read_csv(file_path)

        # 특성과 타겟 데이터 분리
        X = df.drop(['name', 'target'], axis=1)
        y = df['target']

        # 5-fold 교차 검증을 위한 설정
        skf = StratifiedKFold(n_splits=5)

        fold_idx = 1
        y_preds = np.zeros((X.shape[0],))
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model_svc = SVC()

            # RandomizedSearchCV를 사용하여 파라미터 최적화
            param_distributions_svc = {
                'C': uniform(0.1, 10),
                'gamma': uniform(0.01, 1),
                'kernel': ['linear', 'rbf']
            }
            random_search_svc = RandomizedSearchCV(model_svc, param_distributions_svc, n_iter=10, scoring='accuracy', random_state=42)
            random_search_svc.fit(X_train, y_train)

            # 최적의 파라미터 저장
            best_params_svc = random_search_svc.best_params_
            best_params_list_svc.append(best_params_svc)

            # 최적의 파라미터로 모델 학습
            best_model_svc = SVC(**best_params_svc)
            best_model_svc.fit(X_train, y_train)

            # 모델 가중치를 pkl 파일로 저장
            weight_file_path = f"./save/best_model_svc_{file_name.replace('.csv', '')}_fold_{fold_idx}.pkl"
            joblib.dump(best_model_svc, weight_file_path)

            # 예측 및 성능 평가
            y_pred = best_model_svc.predict(X_test)
            y_preds[test_index] = y_pred

            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # MCC 값 계산
            mcc_svc = matthews_corrcoef(y_test, y_pred)
            mcc_values_svc.append(mcc_svc)

            # Accuracy 값 계산
            accuracy_svc = accuracy_score(y_test, y_pred)
            accuracy_values_svc.append(accuracy_svc)

            # Specificity 계산
            specificity = tn / (tn + fp)
            specificity_values_svc.append(specificity)

            # Sensitivity 계산
            sensitivity = tp / (tp + fn)
            sensitivity_values_svc.append(sensitivity)

            print(f"파일 '{file_path}', Fold {fold_idx} MCC (SVC): {mcc_svc}, Accuracy (SVC): {accuracy_svc}, Specificity: {specificity}, Sensitivity: {sensitivity}")

            fold_idx += 1

        # 전체 예측 결과로 메트릭 계산
        cm_ensemble = confusion_matrix(y, y_preds)
        tn_ensemble, fp_ensemble, fn_ensemble, tp_ensemble = cm_ensemble.ravel()

        # 전체 MCC 값 계산
        mcc_ensemble = matthews_corrcoef(y, y_preds)
        # 전체 Accuracy 값 계산
        accuracy_ensemble = accuracy_score(y, y_preds)
        # 전체 Specificity 계산
        specificity_ensemble = tn_ensemble / (tn_ensemble + fp_ensemble)
        # 전체 Sensitivity 계산
        sensitivity_ensemble = tp_ensemble / (tp_ensemble + fn_ensemble)

        print(f"파일 '{file_path}' 앙상블 결과 MCC (SVC): {mcc_ensemble}, Accuracy (SVC): {accuracy_ensemble}, Specificity: {specificity_ensemble}, Sensitivity: {sensitivity_ensemble}")

        # 최종 메트릭 저장
        with open(f"./save/ensemble_result_svc_{file_name.replace('.csv', '')}.txt", 'w') as f:
            f.write(f"Dataset: {file_name}\n")
            f.write(f"Ensemble MCC: {mcc_ensemble}\n")
            f.write(f"Ensemble Accuracy: {accuracy_ensemble}\n")
            f.write(f"Ensemble Specificity: {specificity_ensemble}\n")
            f.write(f"Ensemble Sensitivity: {sensitivity_ensemble}\n\n")
            
    except Exception as e:
        print(f"파일 '{file_path}' 처리 중 오류 발생: {e}")

# 결과를 txt 파일에 저장
output_file_path = "./save/svc_combi_ACP.txt"
with open(output_file_path, 'w') as f:
    for dataset, params_svc, mcc_svc, accuracy_svc, specificity_svc, sensitivity_svc in zip(dataset_names, best_params_list_svc, mcc_values_svc, accuracy_values_svc, specificity_values_svc, sensitivity_values_svc):
        f.write(f"Dataset: {dataset}\nParams: {params_svc}\nMCC: {mcc_svc}\nAccuracy: {accuracy_svc}\nSpecificity: {specificity_svc}\nSensitivity: {sensitivity_svc}\n\n")

print(f"MCC, Accuracy, Specificity, Sensitivity 값 및 최적 파라미터를 '{output_file_path}'에 저장하였습니다.")
