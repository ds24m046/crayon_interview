import os
import sys
import json
import tqdm
import numpy as np
import pandas as pd
from src.classifier.data import cleaners
from src.classifier.data import validators
from sklearn.model_selection import train_test_split


def read_dataset(path):
    
    data = pd.read_csv(path)
    data = data.dropna()

    if not 'product' in data.columns:
        raise ValueError('Data must contain column "product"!')
    
    if not 'narrative' in data.columns:
        raise ValueError('Data must contain column "narrative"!')
    
    if data.shape[0] <= 1000:
        raise ValueError('Data must contain at least 1000 records to run retraining!')
    
    X = list(data['narrative'])
    y = list(data['product'])
    
    return list(zip(X, y))

def prepare_dataset(data):
    
    X = []
    y = []
    errors = []
    
    for n, (email, label) in enumerate(tqdm.tqdm(data)):
        
        email = cleaners.remove_bad_words(email)
        email = cleaners.stem_words(email)
        
        is_valid_email = validators.check_word_count(email)
        
        if is_valid_email:
            X += [email]
            y += [label]
        else:
            errors += [n]
    
    # TODO: log errors
    
    return X, y
    

input_path = 'inputs'
output_path = "artifacts"
test_size = 0.20

if __name__ == "__main__":
    
    if output_path is None:
        raise ValueError('Save path must be specified!')

    if not os.path.exists(output_path):
        raise ValueError('Save path does not exist!')

    dataset = read_dataset(os.path.join(input_path, 'data.csv'))
    X, y = prepare_dataset(dataset)

    for email in X:
        validators.validate_format(email)

    for label in y:
        validators.validate_format(label)

    tag2idx = {i: n for n, i in enumerate(set(y))}
    y = [tag2idx[i] for i in y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

    with open(os.path.join(output_path, 'data.json'), 'w') as file:
        json.dump(data, file)

    with open(os.path.join(output_path, 'tag2idx.json'), 'w') as file:
        json.dump(tag2idx, file)
