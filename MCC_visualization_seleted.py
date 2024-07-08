import glob
import re
import matplotlib.pyplot as plt
import numpy as np

# 사용할 데이터셋 이름 지정
dataset_names = ['AAC.csv', 'DDE.csv', 'DDE_CTDD_CTDC_AAC_DPC.csv',
                 'DDE_AAC.csv', 'DDE_AAC_DPC.csv']

# Accuracy 값을 저장할 딕셔너리
mcc_dict = {dataset: [] for dataset in dataset_names}

# 현재 디렉토리에서 Accuracy가 있는 txt 파일 가져오기
txt_files = glob.glob("./final_selected_feature_model/merged*.txt")

# 각 txt 파일에 대해 작업 수행
for file_name in txt_files:
    with open(file_name, 'r') as f:
        lines = f.readlines()
        current_dataset = None  # 데이터셋 이름 초기화
        for line in lines:
            if line.startswith("Dataset"):
                dataset_match = re.search(r'Dataset: (.+).csv', line)
                if dataset_match:
                    current_dataset = dataset_match.group(1) + '.csv'
            elif line.startswith("Params") and current_dataset in dataset_names:
                mcc_match = re.search(r'MCC: ([\d.]+)', line)
                if mcc_match:
                    mcc = float(mcc_match.group(1))
                    mcc_dict[current_dataset].append(mcc)

# 데이터 정리
model_names = [re.search(r'./final_selected_feature_model/merged_(.+)\.txt', file_name).group(1) for file_name in txt_files]

datasets = list(mcc_dict.keys())
mcc_values = np.zeros((len(datasets), len(model_names)))

for i in range(len(datasets)):
    for j in range(len(model_names)):
        mcc_values[i, j] = mcc_dict[datasets[i]][j] if j < len(mcc_dict[datasets[i]]) else 0

print(mcc_values)

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
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
