import glob
import re
import matplotlib.pyplot as plt
import numpy as np

# Accuracy 값을 저장할 딕셔너리
mcc_dict = {}

# 현재 디렉토리에서 MCC 있는 txt 파일 가져오기
txt_files = glob.glob("./cv_seleted_feature/*_combi_ACP.txt")

# 각 txt 파일에 대해 작업 수행
for file_name in txt_files:
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Dataset"):
                dataset_name = re.search(r'Dataset: (.+).csv', line).group(1)
            elif line.startswith("MCC"):
                mcc = float(re.search(r'MCC: (.+)', line).group(1))
                if dataset_name not in mcc_dict:
                    mcc_dict[dataset_name] = []
                mcc_dict[dataset_name].append(mcc)

print(mcc_dict)
# 데이터 정리
model_names = [re.search(r'./cv_seleted_feature/(.+?)_combi_ACP', file_name).group(1) for file_name in txt_files]
datasets = list(mcc_dict.keys())
mcc_values = np.zeros((len(datasets), len(model_names)))
for i in range(len(datasets)):
    for j in range(len(model_names)):
        mcc_values[i, j] = mcc_dict[datasets[i]][j]


# Heatmap 그리기
plt.figure(figsize=(10, 8))
plt.imshow(mcc_values, cmap='viridis', aspect='auto')

# Annotate heatmap with values
for i in range(len(datasets)):
    for j in range(len(model_names)):
        plt.text(j, i, f'{mcc_values[i, j]:.2f}', ha='center', va='center', color='white')

plt.colorbar(label='MCC Values')
plt.xlabel('Model Names')
plt.ylabel('Datasets')
plt.xticks(np.arange(len(model_names)), model_names, rotation=45)
plt.yticks(np.arange(len(datasets)), datasets)
plt.title('MCC Values for Different Models and Datasets')
plt.tight_layout()
plt.show()