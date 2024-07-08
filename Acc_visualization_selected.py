import glob
import re
import matplotlib.pyplot as plt
import numpy as np

# 사용할 데이터셋 이름 지정
dataset_names = ['CTDD.csv', 'DPC type 2.csv', 'CKSAAP type 2.csv',
                 'ASDC.csv', 'AAC.csv', 'QSOrder.csv']

# Accuracy 값을 저장할 딕셔너리
accuracy_dict = {dataset: [] for dataset in dataset_names}

# 현재 디렉토리에서 Accuracy가 있는 txt 파일 가져오기
txt_files = glob.glob("./resualt/*_feature_ACP.txt")

# 각 txt 파일에 대해 작업 수행

for file_name in txt_files:
    with open(file_name, 'r') as f:
        lines = f.readlines()
        current_dataset = None
        for line in lines:
            if line.startswith("Dataset"):
                dataset_match = re.search(r'Dataset: (.+).csv', line)
                
                if dataset_match:
                    current_dataset = dataset_match.group(1) + '.csv'
            elif line.startswith("Accuracy") and current_dataset in dataset_names:
                accuracy_match = re.search(r'Accuracy: (.+)', line)
                if accuracy_match:
                    accuracy = float(accuracy_match.group(1))
                    accuracy_dict[current_dataset].append(accuracy)

print(accuracy_dict)
# 데이터 정리
model_names = [re.search(r'./resualt/(.+)_feature_ACP.txt',file_name).group(1) for file_name in txt_files]

datasets = list(accuracy_dict.keys())
accuracy_values = np.zeros((len(datasets), len(model_names)))

for i in range(len(datasets)):
    for j in range(len(model_names)):
        accuracy_values[i, j] = accuracy_dict[datasets[i]][j] if j < len(accuracy_dict[datasets[i]]) else 0

print(accuracy_values)

# Heatmap 그리기
plt.figure(figsize=(10, 8))
plt.imshow(accuracy_values, cmap='viridis', aspect='auto')

# Annotate heatmap with values
for i in range(len(datasets)):
    for j in range(len(model_names)):
        plt.text(j, i, f'{accuracy_values[i, j]:.2f}', ha='center', va='center', color='white')

plt.colorbar(label='Accuracy Values')
plt.xlabel('Model Names')
plt.ylabel('Datasets')
plt.xticks(np.arange(len(model_names)), model_names, rotation=45)
plt.yticks(np.arange(len(datasets)), datasets)
plt.title('Accuracy Values for Different Models and Datasets')
plt.tight_layout()
plt.show()
