import sys
import torch
from ..data import cleaners
from ..data import validators


class Predictor:
    
    @classmethod
    def default(self):
        
        return {'label': 'unknown', 'proba': -1.0}
    
    def __init__(self, model, vectorizer, tag2idx):
        
        if torch.cuda.is_available():
            model.to('cuda')
        
        self.model = model.eval()
        self.vectorizer = vectorizer
        self.idx2tag = {val: key for key, val in tag2idx.items()}

    def _predict(self, X):
        
        X = torch.tensor(X, dtype=torch.float32, requires_grad=False)
        
        if torch.cuda.is_available():
            X = X.to('cuda')
        
        with torch.no_grad():
            logits = self.model(X).squeeze()
            probas = torch.sigmoid(logits).numpy()
        
        return probas


class SinglePredictor(Predictor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, email):
        
        X = self.vectorizer.transform([email])
        X = X.todense()
        
        predicted = self._predict(X)
        
        index = predicted.argmax()
        proba = predicted.max()
        
        y_pred = {'label': self.idx2tag[index], 'proba': float(proba)}
        
        return y_pred


class BatchPredictor(Predictor):
    
    def __init__(self):
        super().__init__()

    def predict(self, emails):
        
        X = self.vectorizer.transform(emails)
        X = X.todense()
        
        predicted = self._predict(X)
        
        indices = predicted.argmax(axis=1)
        probas = predicted.max(axis=1)
        
        y_preds = []
        for index, proba in list(zip(indices, probas)):
            y_preds += [{'label': self.idx2tag[index], 'proba': float(proba)}]
        
        return y_preds



