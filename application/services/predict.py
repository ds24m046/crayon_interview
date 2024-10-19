import os
import sys
import json
import torch
import pickle
import functools
sys.path.insert(0, '....')
from pymongo import MongoClient
from datetime import datetime as dt
from classifier.data import cleaners
from classifier.data import validators
from classifier.model import classifiers
from classifier.inference import predictors


def save(email, collection):
    
    record = {
        'timestamp': dt.now().isoformat(),
        'email': email
    }

    result = collection.insert_one(record)
    
    return

def predict(email, predictor=None):
    
    try:
        validators.validate_format(email)
    except Exception as e:
        #TODO: log
        return predictors.Predictor.default()

    email = cleaners.remove_bad_words(email)
    email = cleaners.stem_words(email)

    if not validators.check_word_count(email):
        # TODO: log
        return predictors.Predictor.default()

    return predictor.predict(email)

artifacts_path = '../artifacts'
mongo_path = 'mongodb://localhost:27017'

with open(os.path.join(artifacts_path, 'model_config.json'), 'r') as file:
    config = json.load(file)

with open(os.path.join(artifacts_path, 'vectorizer.pkl'), 'rb') as file:
    vectorizer = pickle.load(file)
    
with open(os.path.join(artifacts_path, 'tag2idx.json'), 'r') as file:
    tag2idx = json.load(file)

state = torch.load(os.path.join(artifacts_path, 'model.pt'), weights_only=True)
model = classifiers.NNClassifier(**config)
model.load_state_dict(state)

predictor = predictors.SinglePredictor(model, vectorizer, tag2idx)

mongo_client = MongoClient(mongo_path)
database = mongo_client['emails']
collection = database['emails']

predict = functools.partial(predict, predictor=predictor)
save = functools.partial(save, collection=collection)
