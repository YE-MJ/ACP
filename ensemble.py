import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from save.model_test.catboost_model import catboost_ensemble
from save.model_test.GB_model import gb_ensemble
from save.model_test.lightgbm_model import lightgbm_ensemble
from save.model_test.RF_model import RF_ensemble
from save.model_test.xgboost_model import xgboost_ensemble
from save.model_test.catboost_model_embedding import catboost_ensemble_enbedding
from save.model_test.SVM_model_embedding import SVM_ensemble_enbedding
from ESM2.test.LSTM_test_ensemble import ensemble_esm_LSTM_test
from ESM2.test.LSTM_independent_ensemble import ensemble_esm_LSTM_independent
from Prott5.test.LSTM_test_ensemble import ensemble_prott5_LSTM_test
from Prott5.test.LSTM_independent_ensemble import ensemble_prott5_LSTM_independent

catboost_proba , y_test = catboost_ensemble()
gb_proba = gb_ensemble()
#lightgbm_proba = lightgbm_ensemble()
RF_proba = RF_ensemble()
xgboost_proba = xgboost_ensemble()

gb_embedding_proba = catboost_ensemble_enbedding()
SVM_embedding_proba = SVM_ensemble_enbedding()

LSTM_esm2_test = ensemble_esm_LSTM_test()
LSTM_prott5_test = ensemble_prott5_LSTM_test()

LSTM_esm2_independent = ensemble_esm_LSTM_independent()
LSTM_prott5_independent = ensemble_prott5_LSTM_independent()


ensemble_proba = np.mean([catboost_proba,gb_proba, RF_proba ,gb_embedding_proba, SVM_embedding_proba, xgboost_proba, LSTM_esm2_test, LSTM_prott5_test], axis=0)

#ensemble_proba = np.mean([catboost_proba,gb_proba, RF_proba ,gb_embedding_proba, SVM_embedding_proba, xgboost_proba, LSTM_esm2_independent, LSTM_prott5_independent], axis=0)


# 최종 예측 (threshold 0.5)
ensemble_pred = (ensemble_proba >= 0.5).astype(int)

# 성능 지표 계산
accuracy = accuracy_score(y_test, ensemble_pred)
roc_auc = roc_auc_score(y_test, ensemble_proba)
f1 = f1_score(y_test, ensemble_pred)
precision = precision_score(y_test, ensemble_pred)
recall = recall_score(y_test, ensemble_pred)
conf_matrix = confusion_matrix(y_test, ensemble_pred)

# Specificity 계산
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)

# 결과 출력
print(f"Accuracy: {accuracy}")
print(f"ROC AUC: {roc_auc}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")
print(f"Confusion Matrix:\n{conf_matrix}")