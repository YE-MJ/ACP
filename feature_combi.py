import pandas as pd
from itertools import combinations
import os

# 파일 경로
file_paths = [
    "./feature/csv/train/DPC type 2.csv",
    "./feature/csv/train/CKSAAP type 2.csv",
    "./feature/csv/train/AAC.csv"
]

# "./feature/csv/train/AAC.csv"
# "./feature/csv/train/QSOrder.csv"

# 파일명에서 확장자 및 경로 제거하고, basename만 추출
file_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in file_paths]

# 데이터프레임 로드
dfs = [pd.read_csv(file_path) for file_path in file_paths]

# 폴더 생성
output_folder = "./cv_seleted_feature/"
os.makedirs(output_folder, exist_ok=True)

# 모든 조합 생성하고 csv로 저장
for r in range(2, len(dfs) + 1):  # 2개 이상의 파일 조합
    for combination in combinations(range(len(dfs)), r):
        combined_df = pd.concat([dfs[idx].drop(columns=['name', 'target']) for idx in combination], axis=1)
        combined_df.insert(0, 'name', dfs[combination[0]]['name'])
        combined_df['target'] = dfs[combination[0]]['target']  # 모든 데이터프레임의 'name'과 'target' 컬럼은 같다고 가정합니다.
        
        combined_file_name = '_'.join([file_names[idx] for idx in combination]) + ".csv"
        combined_file_path = os.path.join(output_folder, combined_file_name)
        combined_df.to_csv(combined_file_path, index=False)
        print(f"Combined result saved as {combined_file_path}")