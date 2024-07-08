import os
import pandas as pd

# CSV 파일이 있는 폴더 경로
folder_path = './feature/csv/test/'

# 폴더 내 모든 파일 확인
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        
        # CSV 파일 읽기
        df = pd.read_csv(file_path)
        
        # 헤더의 맨 앞에 name 열 추가
        df.columns = ['name'] + list(df.columns[1:])
        
        # target 열 추가
        df['target'] = df['name'].apply(lambda x: 1 if 'positive' in x else 0)
        
        # 수정된 DataFrame을 동일한 파일 이름으로 저장 (덮어쓰기)
        df.to_csv(file_path, index=False)

print("CSV 파일 변환이 완료되었습니다.")
