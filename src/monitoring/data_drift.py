import os
import pickle
import numpy as np
from pymongo import MongoClient
from scipy.special import kl_div


def detect_drift(X, expected, thresh=0.5):
    
    distribution = np.asarray(X.sum(axis=0)).squeeze()
    distribution = distribution / X.sum()

    divergence = kl_div(distribution, expected).mean()
    
    return divergence > thresh


if __name__ == '__main__':
    
    limit = 1000
    artifacts_path = 'artifacts'
    mongo_path = os.getenv('MONGO_URI', 'mongodb://localhost:27017/emails')

    client = MongoClient(mongo_path)
    collection = client['emails']['emails']

    with open(os.path.join(artifacts_path, 'distribution.pkl'), 'rb') as file:
        expected = pickle.load(file)

    with open(os.path.join(artifacts_path, 'vectorizer.pkl'), 'rb') as file:
        vectorizer = pickle.load(file)

    emails = list(collection.find().limit(limit))
    emails = [i['email'] for i in emails]
        
    X = vectorizer.transform(emails)

    is_drift = detect_drift(X, expected, thresh=0.5)
    print(is_drift)