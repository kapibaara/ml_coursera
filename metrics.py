import pandas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import numpy as np

data = pandas.read_csv('data_seta/scores.csv')

y_true = data['true'].to_numpy()
y_pred = data['pred'].to_numpy()

TP = 0
TN = 0
FP = 0
FN = 0

for i in range(0, len(y_true)):
    if y_true[i] == 1 and y_pred[i] == 1:
        TP += 1
    if y_true[i] == 0 and y_pred[i] == 1:
        FP += 1
    if y_true[i] == 1 and y_pred[i] == 0:
        FN += 1
    if y_true[i] ==0 and y_pred[i] == 0:
        TN += 1

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f = f1_score(y_true, y_pred)

classifiers = pandas.read_csv('data_seta/classification.csv')

y1_true = classifiers['true'].to_numpy()
score_logreg = classifiers['score_logreg'].to_numpy()
score_svm = classifiers['score_svm'].to_numpy()
score_knn = classifiers['score_knn'].to_numpy()
score_tree = classifiers['score_tree'].to_numpy()

logreg = roc_auc_score(y1_true, score_logreg)
svm = roc_auc_score(y1_true, score_svm)
knn = roc_auc_score(y1_true, score_knn)
tree = roc_auc_score(y1_true, score_tree)

prc_logred = precision_recall_curve(y1_true, score_logreg)
prc_svm = precision_recall_curve(y1_true, score_svm)
prc_knn = precision_recall_curve(y1_true, score_knn)
prc_tree = precision_recall_curve(y1_true, score_tree)

def find_biggest_precision_when_recall(array):
    precision, recall, thresholds = array

    recall_indx = np.where(recall > 0.7)
    filtered_precision = precision[recall_indx]

    return round(max(filtered_precision), 2)


print(find_biggest_precision_when_recall(prc_logred))
print(find_biggest_precision_when_recall(prc_svm))
print(find_biggest_precision_when_recall(prc_knn))
print(find_biggest_precision_when_recall(prc_tree))