import numpy as np
import tensorflow as tf

def custom_metrics(confusion_matrix):
    '''
    Compute:
        - precision, recall and f1 per class
        - accuracy
        - precision, recall and f1 macro
    Return:
        - dict with a liste for precision, recall and f1
    '''
    confusion_matrix = confusion_matrix.numpy()
    num_user = confusion_matrix.shape[0]
    metrics = {
        'accuracy': 0,
        'precision': [],
        'recall': [],
        'f1': [],
        'macro_precision': 0,
        'macro_recall': 0,
        'macro_f1': 0
    }

    for user in range(0,num_user):
        tp = confusion_matrix[user,user]
        precision = tp / np.sum(confusion_matrix[:,user])
        recall = tp / np.sum(confusion_matrix[user,:])
        f1 = 2 * precision * recall / (precision + recall)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
    
    metrics['macro_precision'] = np.sum(metrics['precision']) / len(metrics['precision'])
    metrics['macro_recall'] = np.sum(metrics['recall']) / len(metrics['recall'])
    metrics['macro_f1'] = np.sum(metrics['f1']) / len(metrics['f1'])
    metrics['accuracy'] = np.sum(confusion_matrix.diagonal()) / np.sum(confusion_matrix[:,:])

    return metrics