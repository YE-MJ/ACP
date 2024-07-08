import os
import pandas as pd

# preprocessing_feature_extraction 디렉토리 경로 설정
input_directory_path = "./feature/csv/train/"

# 디렉토리 내의 모든 CSV 파일 경로 가져오기
csv_files = [f for f in os.listdir(input_directory_path) if f.endswith(".csv")]

# 각 CSV 파일에 대해 작업 수행
for file_name in csv_files:
    file_path = os.path.join(input_directory_path, file_name)
    
    # CSV 파일 불러오기
    df = pd.read_csv(file_path)

    # NaN 값이 있는지 확인
    if df.isnull().values.any():
        print(f"File '{file_name}' contains NaN values.")