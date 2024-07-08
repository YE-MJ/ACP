import torch
import numpy as np
import joblib
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from model import SequenceModel_deep
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from utils import calculate_metrics, calculate_additional_metrics, load_data

# Inde Test
new_data_file_path = './Dataset/independent/protein_embeddings_prott5.csv'
new_embeddings_df = pd.read_csv(new_data_file_path)

# Name, Target 제외 
X_prott5_new = new_embeddings_df.iloc[:, 1:-1].values  
y_new = new_embeddings_df['target'].values

positive_file_path = './Dataset/independent_test_data/independent_ACP_data(positive).txt'
negative_file_path = './Dataset/independent_test_data/independent_ACP_data(negative).txt'

positive_sequences_new = load_data(positive_file_path)
negative_sequences_new = load_data(negative_file_path)
sequences_new = positive_sequences_new + negative_sequences_new

def calculate_binary_profile_features(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_binary = {aa: np.array([1 if amino_acids[i] == aa else 0 for i in range(len(amino_acids))]) for aa in amino_acids}
    binary_profile = [aa_to_binary.get(aa, np.zeros(20)) for aa in sequence]
    return np.array(binary_profile).reshape(-1)

bpf_features_new = [calculate_binary_profile_features(seq) for seq in sequences_new]
max_bpf_length_new = max(len(features) for features in bpf_features_new)
bpf_features_padded_new = np.array([np.pad(features, (0, max_bpf_length_new - len(features)), 'constant') for features in bpf_features_new])

X_combined_new = np.hstack((X_prott5_new, bpf_features_padded_new))

scaler = joblib.load('./Prott5/LSTM(deep)_Prott5+BPF_weights/scaler_LSTM(deep)_Prott5+BPF.pkl')
X_combined_scaled_new = scaler.fit_transform(X_combined_new)

X_tensor_new = torch.tensor(X_combined_scaled_new, dtype=torch.float)
y_tensor_new = torch.tensor(y_new, dtype=torch.long)

# Test 
best_model_paths_acc = [
    './Prott5/LSTM(deep)_Prott5+BPF_weights/best_model_fold_1_acc.pth',
    './Prott5/LSTM(deep)_Prott5+BPF_weights/best_model_fold_2_acc.pth',
    './Prott5/LSTM(deep)_Prott5+BPF_weights/best_model_fold_3_acc.pth',
    './Prott5/LSTM(deep)_Prott5+BPF_weights/best_model_fold_4_acc.pth',
    './Prott5/LSTM(deep)_Prott5+BPF_weights/best_model_fold_5_acc.pth'
]

input_dim = X_combined_scaled_new.shape[1]
hidden_dim = 128
output_dim = 2

# 각 fold별 결과를 저장할 리스트
all_metrics = {
    'accuracy': [],
    'sensitivity': [],
    'specificity': [],
    'mcc': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'roc_auc': []
}

all_y_true = []
all_y_pred = []
all_y_scores = []

# 각 fold별 모델 로드 및 예측 (best - acc)
for model_path in best_model_paths_acc:
    model = SequenceModel_deep(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_dataset = TensorDataset(X_tensor_new, y_tensor_new)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            positive_probabilities = probabilities[:, 1].numpy()

            y_true.extend(labels.numpy()) # 정답값
            y_scores.extend(positive_probabilities) # 확률 
            y_pred.extend(np.where(positive_probabilities > 0.5, 1, 0)) # 0 or 1 변환

    # 성능 지표 계산
    accuracy, sensitivity, specificity, mcc = calculate_metrics(y_true, y_pred)
    precision, recall, f1 = calculate_additional_metrics(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_scores)

    # 각 fold별 성능 지표 저장
    all_metrics['accuracy'].append(accuracy)
    all_metrics['sensitivity'].append(sensitivity)
    all_metrics['specificity'].append(specificity)
    all_metrics['mcc'].append(mcc)
    all_metrics['precision'].append(precision)
    all_metrics['recall'].append(recall)
    all_metrics['f1'].append(f1)
    all_metrics['roc_auc'].append(roc_auc)

    all_y_true.append(y_true)
    all_y_pred.append(y_pred)
    all_y_scores.append(y_scores)

# 각 성능 지표의 평균 계산
average_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}

print(f"Average ACC: {average_metrics['accuracy']*100:.2f}%")
print(f"Average Sen: {average_metrics['sensitivity']*100:.2f}%")
print(f"Average Spc: {average_metrics['specificity']*100:.2f}%")
print(f"Average MCC: {average_metrics['mcc']:.3f}")
print(f"Average Precision: {average_metrics['precision']:.3f}")
print(f"Average Recall: {average_metrics['recall']:.3f}")
print(f"Average F1 Score: {average_metrics['f1']:.3f}")
print(f"Average ROC AUC: {average_metrics['roc_auc']:.3f}")

# 결과 저장
result_df = pd.DataFrame({
    'y_true': np.concatenate(all_y_true),
    'y_score': np.concatenate(all_y_scores),
    'y_pred': np.concatenate(all_y_pred)
})

result_df.to_csv('./Prott5/results/Prott5+BPF_test_results(Inde)_acc.csv', index=False)

# ROC Curve 
plt.figure(figsize=(8, 6))
mean_fpr = np.linspace(0, 1, 100)
tprs = []
for i in range(len(best_model_paths_acc)):
    fpr, tpr, _ = roc_curve(all_y_true[i], all_y_scores[i])
    plt.plot(fpr, tpr, lw=2, label=f'Fold {i+1} ROC curve (area = {all_metrics["roc_auc"][i]:.2f})')
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, color='b', lw=2, linestyle='--', label=f'Mean ROC curve (area = {mean_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 전체 Confusion Matrix 
all_y_true_combined = np.concatenate(all_y_true)
all_y_pred_combined = np.concatenate(all_y_pred)
cm = confusion_matrix(all_y_true_combined, all_y_pred_combined)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Overall Confusion Matrix')
plt.show()

# 각 폴드별 Confusion Matrix 
for i in range(len(best_model_paths_acc)):
    cm = confusion_matrix(all_y_true[i], all_y_pred[i])
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix Fold {i+1}')
    plt.show()