import os
import json
import torch
import scipy
import pickle
import numpy as np
from src.utils import mongo
from src.classifier.model import classifiers
from src.classifier.inference import predictors


def get_data_drift(X, expected):
    
    distribution = np.asarray(X.sum(axis=0)).squeeze()
    distribution = distribution / X.sum()

    divergence = scipy.special.kl_div(distribution, expected)
    
    return divergence.mean()

def get_model_drift(emails, predictor):
    
    y_pred = predictor.predict(emails)
    confidence = [i['proba'] for i in y_pred]
    
    return np.mean(confidence)

def run(input_path='artifacts', limit=1000):
    
    collections = {key: mongo.get_collection(key) for key in ('emails', 'drifts')}
    
    with open(os.path.join(input_path, 'distribution.pkl'), 'rb') as file:
        expected = pickle.load(file)

    with open(os.path.join(input_path, 'vectorizer.pkl'), 'rb') as file:
        vectorizer = pickle.load(file)
    
    with open(os.path.join(input_path, 'model_config.json'), 'r') as file:
        config = json.load(file)

    with open(os.path.join(input_path, 'tag2idx.json'), 'r') as file:
        tag2idx = json.load(file)

    state = torch.load(os.path.join(input_path, 'model.pt'), weights_only=True)
    model = classifiers.NNClassifier(**config)
    model.load_state_dict(state)

    predictor = predictors.BatchPredictor(model, vectorizer, tag2idx)
    
    emails = mongo.get_emails(collections['emails'], limit=1000)    
    X = vectorizer.transform(emails)
    
    data_drift = {'type': 'data', 'metric': get_data_drift(X, expected)}
    mongo.save_drift(collections['drifts'], data_drift)
    
    model_drift = {'type': 'model', 'metric': get_model_drift(emails, predictor)}
    mongo.save_drift(collections['drifts'], model_drift)


if __name__ == "__main__":
    
    limit = 1000
    input_path = 'artifacts'
    
    run(input_path, limit)
