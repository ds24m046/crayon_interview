import os
import json
import torch
import pickle
import numpy as np
from pymongo import MongoClient
from src.classifier.model import classifiers
from src.classifier.inference import predictors


def detect_drift(predictor, emails, thresh=0.5):
    
    y_pred = predictor.predict(emails)
    
    confidence = [i['proba'] for i in y_pred]
    confidence = np.mean(confidence)
    
    return confidence < thresh


if __name__ == '__main__':
    
    limit = 1000
    artifacts_path = 'artifacts'
    mongo_path = os.getenv('MONGO_URI', 'mongodb://localhost:27017/emails')

    client = MongoClient(mongo_path)
    collection = client['emails']['emails']

    with open(os.path.join(artifacts_path, 'model_config.json'), 'r') as file:
        config = json.load(file)

    with open(os.path.join(artifacts_path, 'vectorizer.pkl'), 'rb') as file:
        vectorizer = pickle.load(file)
    
    with open(os.path.join(artifacts_path, 'tag2idx.json'), 'r') as file:
        tag2idx = json.load(file)

    state = torch.load(os.path.join(artifacts_path, 'model.pt'), weights_only=True)
    model = classifiers.NNClassifier(**config)
    model.load_state_dict(state)

    predictor = predictors.BatchPredictor(model, vectorizer, tag2idx)

    emails = list(collection.find().limit(limit))
    emails = [i['email'] for i in emails]
    
    is_drift = detect_drift(predictor, emails, thresh=0.5)
    print(is_drift)