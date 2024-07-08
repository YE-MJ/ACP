def merge_files(file1, file2, output_file):
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w') as out_f:
        # file1의 내용을 output 파일로 복사
        for line in f1:
            out_f.write(line)
        
        # file2의 내용을 output 파일로 복사
        for line in f2:
            out_f.write(line)

# 입력 파일 및 출력 파일 지정
file1 = './ACP_data/antiCP2_positive.txt' # 첫 번째 파일명을 실제 파일명으로 변경해주세요.
file2 = './ACP_data/antiCP2_negative.txt'  # 두 번째 파일명을 실제 파일명으로 변경해주세요.
output_file = './ACP_data.txt' # 병합된 파일명을 원하는 이름으로 변경해주세요.

# 두 파일 병합
merge_files(file1, file2, output_file)