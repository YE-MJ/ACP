import os
import pandas as pd
import joblib
import numpy as np

def catboost_ensemble():
    # 테스트할 CSV 파일 경로 설정
    test_file_path = './feature/csv/test/'   # 여기에 테스트 파일 경로를 입력해주세요
    test_files = [f for f in os.listdir(test_file_path) if f.endswith('.csv')]

    # 사용할 CSV 파일 경로 리스트 설정
    csv_files = [
        './selected_feature/AAC.csv', './selected_feature/CKSAAP type 2.csv',
        './selected_feature/DPC tpye 2.csv', './selected_feature/CKSAAP type 2_AAC.csv',
        './selected_feature/DPC tpye 2_AAC.csv', './selected_feature/CKSAAP type 2_DPC tpye 2.csv',
        './selected_feature/DPC tpye 2_CKSAAP type 2_AAC.csv'
    ]

    # csv_files 리스트에서 파일 이름만 추출
    csv_file_names = [os.path.basename(f) for f in csv_files]

    all_predictions = []
    y_tests = []

    for test_file in test_files:
        # test_file이 csv_file_names에 있는지 확인
        if test_file not in csv_file_names:
            continue
        
        test_df = pd.read_csv(os.path.join(test_file_path, test_file))

        # 특성 데이터 추출 (여기서는 'name' 컬럼은 필요 없다고 가정)
        X_test = test_df.drop(['name', 'target'], axis=1)
        y_test = test_df['target']

        y_tests.append(y_test)
        
        tmp = []

        # 각 데이터셋 파일명에 대해 작업 수행
        for file_path in csv_files:
            if os.path.basename(file_path) != test_file:
                continue
            
            file_name = os.path.basename(file_path)
            
            # 모델 가중치 파일 경로 설정
            weight_file_paths = [
                f"./selected/best_model_catboost_{file_name.replace('.csv', '')}_fold_1.pkl",
                f"./selected/best_model_catboost_{file_name.replace('.csv', '')}_fold_2.pkl",
                f"./selected/best_model_catboost_{file_name.replace('.csv', '')}_fold_3.pkl",
                f"./selected/best_model_catboost_{file_name.replace('.csv', '')}_fold_4.pkl",
                f"./selected/best_model_catboost_{file_name.replace('.csv', '')}_fold_5.pkl"
            ]

            # 각 fold에서 예측값을 저장할 리스트
            y_preds = []

            # 각 fold의 모델 가중치를 불러와서 예측 수행
            for weight_file_path in weight_file_paths:
                # 모델 가중치 로드
                model = joblib.load(weight_file_path)

                # 예측 수행
                y_pred = model.predict(X_test)
                y_preds.append(y_pred)

            # 각 fold에서의 예측값을 평균하여 최종 예측값 계산 (앙상블)
            y_preds_ensemble = np.mean(y_preds, axis=0)
            tmp.append(y_preds_ensemble)

        all_predictions.append(np.mean(tmp, axis=0))
        
    final_prediction = np.mean(all_predictions, axis=0)
    
    return final_prediction, y_test

