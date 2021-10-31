import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_recall_fscore_support

def cal_acc_auc(label, pred, prob):
    acc = accuracy_score(label, pred)
    if not np.any(label == 1) or not np.any(label == 0):
        return acc, 0.
    auc = roc_auc_score(label, pred)
    return acc, auc 

def cal_confusion_matrix(label, pred):
    tn, fp, fn, tp = confusion_matrix(y_true=label, y_pred=pred, labels=[0,1]).ravel() 
    return tn, fp, fn, tp

def cal_fscore(label, pred):
   prc, rcl, f1, _ = precision_recall_fscore_support(y_true=label, y_pred=pred, average='weighted')
   return prc, rcl, f1