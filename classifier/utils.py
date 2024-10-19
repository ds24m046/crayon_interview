import tqdm
import json
import numpy as np
from classifier.data import cleaners
from classifier.data import validators


def create_label_mapping(data):
    
    labels = [label for _, label in data]
    tag2idx = {i: n for n, i in enumerate(set(labels))}
    
    return tag2idx
    
def prepare_data(data, tag2idx):
    
    dataset = {'X': [], 'y': []}
    errors = {'bad_email': [], 'bad_label': []}
    
    for n, (email, label) in enumerate(tqdm.tqdm(data)):
        
        email = cleaners.remove_bad_words(email)
        email = cleaners.stem_words(email)
        
        is_valid_email = validators.check_word_count(email)
        is_valid_label = validators.check_label_existance(label, tag2idx)
        
        if is_valid_email and is_valid_label:
            dataset['X'] += [email]
            dataset['y'] += [label]
        elif is_valid_email and not is_valid_label:
            errors['bad_label'] += [n]
        elif not is_valid_email and is_valid_label:
            errors['bad_email'] += [n]
        else:
            errors['bad_email'] += [n]
            errors['bad_label'] += [n]

    dataset['y'] = np.array([tag2idx[i] for i in dataset['y']])
    
    return dataset, errors
    
    