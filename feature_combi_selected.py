import pandas as pd
from itertools import combinations
import os

# 파일 경로
file_paths = [
    "./selected_feature/AAC.csv",
    "./selected_feature/CKSAAP type 2.csv",
    "./selected_feature/DPC type 2.csv",
]

# 파일명에서 확장자 및 경로 제거하고, basename만 추출
file_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in file_paths]

# 데이터프레임 로드
dfs = {file_name: pd.read_csv(file_path) for file_name, file_path in zip(file_names, file_paths)}


# 폴더 생성
output_folder = "./selected_feature/"
os.makedirs(output_folder, exist_ok=True)

# 원하는 조합 리스트
desired_combinations = [
    ["DPC type 2", "AAC"],
    ["CKSAAP type 2", "AAC"],
    ["CKSAAP type 2", "DPC type 2"],
    ["DPC type 2", "CKSAAP type 2", "AAC"]
]

# 데이터프레임 결합 및 CSV 저장
for combo in desired_combinations:
    # 결합할 데이터프레임 선택
    combined_df = pd.concat([dfs[name].drop(columns=['name', 'target']) for name in combo], axis=1)
    combined_df.insert(0, 'name', dfs[combo[0]]['name'])
    combined_df['target'] = dfs[combo[0]]['target'] 
    # 파일명 생성
    combo_name = "_".join(combo)
    output_file = os.path.join(output_folder, f"{combo_name}.csv")
    # CSV 저장
    combined_df.to_csv(output_file, index=False)
    print(f"Saved {output_file}")