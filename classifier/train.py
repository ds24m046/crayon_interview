import os
import sys
import json
import torch
import pickle
import argparse
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, '..')
import utils
from classifier.data import validators
from classifier.training import datasets
from classifier.training import trainers
from classifier.model import classifiers


def train(data, n_epochs=2, batch_size=256, learning_rate=0.001, validation_size = 0.20, save_path=None):

    if save_path is None:
        raise ValueError('Save path must be specified!')
    
    # Validate inputs
    # TODO: validate others
    
    if not os.path.exists(save_path):
        raise ValueError('Save path does not exist!')
    
    if not isinstance(data, list) or not all(isinstance(i, tuple) for i in data) or not all(len(i)==2 for i in data):
        raise ValueError('Data format must be list of (email, label) tuples!')
    
    for email, label in data:
        validators.validate_format(email)
        validators.validate_format(label)

    # Prepare dataset
    # TODO: log errors
    
    print('Preparing dataset...')

    tag2idx = utils.create_label_mapping(data)
    data, errors = utils.prepare_data(data, tag2idx)

    X_train, X_val, y_train, y_val = train_test_split(data['X'], data['y'], test_size=validation_size, random_state=0)
    
    # Vectorize inputs

    print('Vectorizing dataset...')

    vectorizer = TfidfVectorizer(min_df=5, max_df=0.75, analyzer='word', stop_words='english')
    vectorizer.fit(X_train)

    encoder = OneHotEncoder()
    encoder.fit(y_train.reshape(-1, 1))

    train_dataset = datasets.Dataset(X_train, y_train, vectorizer, encoder)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = datasets.Dataset(X_val, y_val, vectorizer, encoder)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train model

    print('Training...')

    config = {
        'n_input': train_dataset.X.shape[1],
        'n_target': len(tag2idx)
    }

    model = classifiers.NNClassifier(**config)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = trainers.Trainer(model, optimizer, loss_fn)
    losses, metrics = trainer.train(train_loader, val_loader, n_epochs=n_epochs)

    # Save artifacts

    with open(os.path.join(save_path, 'tag2idx.json'), 'w') as file:
        json.dump(tag2idx, file)

    with open(os.path.join(save_path, 'config.json'), 'w') as file:
        json.dump(config, file)

    with open(os.path.join(save_path, 'vectorizer.pkl'), 'wb') as file:
        pickle.dump(vectorizer, file)

    torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
