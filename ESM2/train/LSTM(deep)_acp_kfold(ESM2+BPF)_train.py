import torch
import torch.nn.functional as F
from torch import tensor
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import seaborn as sns
import joblib
import numpy as np
from scipy import interp
from model import SequenceModel_deep
import pandas as pd 
from utils import calculate_metrics, calculate_additional_metrics, plot_roc_curve, plot_confusion_matrix, load_data

# 데이터 import 
file_path = './Dataset/train/protein_embeddings_esm2.csv'
embeddings_df = pd.read_csv(file_path)

# Name, Target 제외 
X_esm2 = embeddings_df.iloc[:, 1:-1].values  
y = embeddings_df['target'].values

positive_file_path = './Dataset/main_ACP_data/antiCP2_train_ACP_data(positive).txt'
negative_file_path = './Dataset/main_ACP_data/antiCP2_train_ACP_data(negative).txt'

positive_sequences = load_data(positive_file_path)
negative_sequences = load_data(negative_file_path)
sequences = positive_sequences + negative_sequences

def calculate_binary_profile_features(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_binary = {aa: np.array([1 if amino_acids[i] == aa else 0 for i in range(len(amino_acids))]) for aa in amino_acids}
    binary_profile = [aa_to_binary.get(aa, np.zeros(20)) for aa in sequence]
    return np.array(binary_profile).reshape(-1)  

bpf_features = [calculate_binary_profile_features(seq) for seq in sequences]
max_bpf_length = max(len(features) for features in bpf_features)
bpf_features_padded = np.array([np.pad(features, (0, max_bpf_length - len(features)), 'constant') for features in bpf_features])

X_combined = np.hstack((X_esm2, bpf_features_padded))

scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)
joblib.dump(scaler, './ESM2/LSTM(deep)_ESM2+BPF_weights/scaler_LSTM(deep)_ESM2+BPF.pkl')

X_tensor = torch.tensor(X_combined_scaled, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long) 

# 5fold 
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fprs, tprs, aucs = [], [], []
results = []

for fold, (train_index, test_index) in enumerate(kf.split(X_tensor), 1):
    print(f"Fold {fold}")

    X_train, X_test = X_tensor[train_index], X_tensor[test_index]
    y_train, y_test = y_tensor[train_index], y_tensor[test_index]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = X_combined_scaled.shape[1]
    hidden_dim = 128
    output_dim = 2

    model = SequenceModel_deep(input_dim, hidden_dim, output_dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    epochs = 100

    best_loss = float('inf')
    best_auc = 0.0
    best_acc = 0.0
    best_model_wts = None

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
        
        model.eval()
        y_true = []
        y_scores = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                y_true.extend(labels.numpy())
                y_scores.extend(outputs[:, 1].numpy())
        
        roc_auc = roc_auc_score(y_true, y_scores)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, f'./ESM2/LSTM(deep)_ESM2+BPF_weights/best_model_fold_{fold}_loss.pth')
        
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, f'./ESM2/LSTM(deep)_ESM2+BPF_weights/best_model_fold_{fold}_auc.pth')

    y_true = []
    y_pred = []
    y_scores = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())
            y_scores.extend(outputs[:, 1].numpy())  
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    fprs.append(fpr)
    tprs.append(tpr)
    aucs.append(roc_auc)
    
    acc, sen, spc, mcc = calculate_metrics(y_true, y_pred)
    precision, recall, f1 = calculate_additional_metrics(y_true, y_pred)
    results.append((acc, sen, spc, mcc, precision, recall, f1))
    print(f"Fold {fold} - ACC: {acc*100:.2f}, Sen: {sen*100:.2f}, Spc: {spc*100:.2f}, MCC: {mcc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    if acc > best_acc:
        best_acc = acc
        best_model_wts = model.state_dict()
        torch.save(best_model_wts, f'./ESM2/LSTM(deep)_ESM2+BPF_weights/best_model_fold_{fold}_acc.pth')

    plot_roc_curve(np.array(y_true), np.array(y_scores))    
    plot_confusion_matrix(np.array(y_true), np.array(y_pred))

all_fpr = np.unique(np.concatenate([fprs[i] for i in range(len(fprs))]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(fprs)):
    mean_tpr += interp(all_fpr, fprs[i], tprs[i])
mean_tpr /= len(fprs)
mean_tpr[-1] = 1.0
mean_auc = auc(all_fpr, mean_tpr)

plt.figure(figsize=(8, 6))
for i in range(len(fprs)):
    plt.plot(fprs[i], tprs[i], lw=1, alpha=0.3, label=f'Fold {i+1} (AUC = {aucs[i]:.2f})')

plt.plot(all_fpr, mean_tpr, color='blue', label=f'Mean AUC = {mean_auc:.2f}', lw=2, alpha=0.8)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Folds with Mean AUC')
plt.legend(loc="lower right")
plt.show()

mean_results = np.mean(results, axis=0)
print(f"Mean Results - ACC: {mean_results[0]*100:.2f}%, Sen: {mean_results[1]*100:.2f}%, Spc: {mean_results[2]*100:.2f}%, MCC: {mean_results[3]:.3f}, Precision: {mean_results[4]:.3f}, Recall: {mean_results[5]:.3f}, F1: {mean_results[6]:.3f}")