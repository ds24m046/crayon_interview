import os
import sys
import json
import torch
import pickle
import functools
from src.utils import mongo
from src.classifier.data import cleaners
from src.classifier.data import validators
from src.classifier.model import classifiers
from src.classifier.inference import predictors


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


artifacts_path = 'artifacts'

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
predict = functools.partial(predict, predictor=predictor)

collection = mongo.get_collection('emails')
save = functools.partial(mongo.save_email, collection=collection)
