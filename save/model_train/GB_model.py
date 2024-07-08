import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from scipy.stats import randint
import joblib
import numpy as np

# 사용할 CSV 파일 경로 리스트 설정
csv_files = [
    './cv_seleted_feature/CKSAAP type 2_DPC type 2.csv'
]
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

            model_gb = GradientBoostingClassifier()

            # RandomizedSearchCV를 사용하여 파라미터 최적화
            param_distributions_gb = {
                'n_estimators': randint(50, 500),
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'max_depth': randint(3, 10),
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'max_features': ['auto', 'sqrt', 'log2', None]
            }
            random_search_gb = RandomizedSearchCV(model_gb, param_distributions_gb, n_iter=10, scoring='accuracy', random_state=42)
            random_search_gb.fit(X_train, y_train)

            # 최적의 파라미터 저장
            best_params_gb = random_search_gb.best_params_
            best_params_list_gb.append(best_params_gb)

            # 최적의 파라미터로 모델 학습
            best_model_gb = GradientBoostingClassifier(**best_params_gb)
            best_model_gb.fit(X_train, y_train)

            # 모델 가중치를 pkl 파일로 저장
            weight_file_path = f"./cv_seleted_feature/best_model_gb_{file_name.replace('.csv', '')}_fold_{fold_idx}.pkl"
            joblib.dump(best_model_gb, weight_file_path)

            # 예측 및 성능 평가
            y_pred = best_model_gb.predict(X_test)
            y_preds[test_index] = y_pred

            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # MCC 값 계산
            mcc_gb = matthews_corrcoef(y_test, y_pred)
            mcc_values_gb.append(mcc_gb)

            # Accuracy 값 계산
            accuracy_gb = accuracy_score(y_test, y_pred)
            accuracy_values_gb.append(accuracy_gb)

            # Specificity 계산
            specificity = tn / (tn + fp)
            specificity_values_gb.append(specificity)

            # Sensitivity 계산
            sensitivity = tp / (tp + fn)
            sensitivity_values_gb.append(sensitivity)

            print(f"파일 '{file_path}', Fold {fold_idx} MCC (Gradient Boosting): {mcc_gb}, Accuracy (Gradient Boosting): {accuracy_gb}, Specificity: {specificity}, Sensitivity: {sensitivity}")

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

        print(f"파일 '{file_path}' 앙상블 결과 MCC (Gradient Boosting): {mcc_ensemble}, Accuracy (Gradient Boosting): {accuracy_ensemble}, Specificity: {specificity_ensemble}, Sensitivity: {sensitivity_ensemble}")

        # 최종 메트릭 저장
        with open(f"./cv_seleted_feature/ensemble_result_gb_2_{file_name.replace('.csv', '')}.txt", 'w') as f:
            f.write(f"Dataset: {file_name}\n")
            f.write(f"Ensemble MCC: {mcc_ensemble}\n")
            f.write(f"Ensemble Accuracy: {accuracy_ensemble}\n")
            f.write(f"Ensemble Specificity: {specificity_ensemble}\n")
            f.write(f"Ensemble Sensitivity: {sensitivity_ensemble}\n\n")
    except Exception as e:
        print(f"파일 '{file_path}' 처리 중 오류 발생: {e}")

# 결과를 txt 파일에 저장
output_file_path = "./cv_seleted_feature/gradient_boosting_combi_ACP_2.txt"
with open(output_file_path, 'w') as f:
    for dataset, params_gb, mcc_gb, accuracy_gb, specificity_gb, sensitivity_gb in zip(dataset_names, best_params_list_gb, mcc_values_gb, accuracy_values_gb, specificity_values_gb, sensitivity_values_gb):
        f.write(f"Dataset: {dataset}\nParams: {params_gb}\nMCC: {mcc_gb}\nAccuracy: {accuracy_gb}\nSpecificity: {specificity_gb}\nSensitivity: {sensitivity_gb}\n\n")

print(f"MCC, Accuracy, Specificity, Sensitivity 값 및 최적 파라미터를 '{output_file_path}'에 저장하였습니다.")
