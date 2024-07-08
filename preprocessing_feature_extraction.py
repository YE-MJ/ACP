import pandas as pd
import glob
import os
import csv

# feature_extraction 디렉토리 경로 설정
input_directory_path = "./ACP_feature/txt/"

# preprocessed_feature_extraction 디렉토리 경로 설정
output_directory_path = "./ACP_feature/csv/"

# 디렉토리가 없는 경우 생성
if not os.path.exists(output_directory_path):
    os.makedirs(output_directory_path)

# 디렉토리 내의 모든 TXT 파일 경로 가져오기
txt_files = [f for f in os.listdir(input_directory_path) if f.endswith(".txt")]

# 각 TXT 파일에 대해 작업 수행
for file_name in txt_files:
    file_path = os.path.join(input_directory_path, file_name)

    # 새로운 CSV 파일 경로 설정
    csv_file_path = os.path.join(output_directory_path, os.path.splitext(file_name)[0] + ".csv")

    # CSV 파일로 변환하여 저장
    with open(file_path, 'r') as txt_file, open(csv_file_path, 'w', newline='') as csv_file:
        # CSV 라이터 생성
        csv_writer = csv.writer(csv_file)

        # 첫 번째 행 처리
        first_line = txt_file.readline().strip()
        # 첫 번째 행의 시작이 #인 경우 name으로 바꾸기
        if first_line.startswith("#"):
            first_line = "name" + first_line[1:]
        # CSV 파일의 첫 번째 행에 쓰기
        csv_writer.writerow(first_line.split('\t') + ["target"])

        # 나머지 행에 대해 positive인 경우 1, negative인 경우 0으로 설정하여 CSV 파일에 쓰기
        for line in txt_file:
            parts = line.strip().split('\t')
            target_value = '0' if 'negative' in parts[0] else '1'
            csv_writer.writerow(parts + [target_value])

    print(f"File '{file_path}' converted and saved as '{csv_file_path}'")

print("All files converted to CSV format with 'target' column and saved in 'preprocessed_feature_extraction' directory.")