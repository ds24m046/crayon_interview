
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# TODO:
# classification_report
# confusion_matrix

def evaluate_model(y_true, y_pred):
    
    scores = {
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0.0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0.0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0.0)
    }
    
    return scores
