import os
import sys
import json
import torch
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from src.classifier.training import datasets
from src.classifier.training import trainers
from src.classifier.model import classifiers


n_epochs=1
batch_size=256
learning_rate=0.001
save_path = 'artifacts'
load_path = 'data'

if __name__ == "__main__":

    with open(os.path.join(load_path, 'data.json'), 'r') as file:
        data = json.load(file)

    with open(os.path.join(load_path, 'tag2idx.json'), 'r') as file:
        tag2idx = json.load(file)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = np.array(data['y_train'])
    y_test = np.array(data['y_test'])
    
    print('Vectorizing dataset...')
    # TODO: logging

    vectorizer = TfidfVectorizer(min_df=5, max_df=0.75, analyzer='word', stop_words='english')
    vectorizer.fit(X_train)

    encoder = OneHotEncoder()
    encoder.fit(y_train.reshape(-1, 1))
    
    train_dataset = datasets.Dataset(X_train, y_train, vectorizer, encoder)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = datasets.Dataset(X_test, y_test, vectorizer, encoder)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print('Training...')
    # TODO: logging

    config = {
        'n_input': train_dataset.X.shape[1],
        'n_target': len(tag2idx)
    }

    model = classifiers.NNClassifier(**config)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = trainers.Trainer(model, optimizer, loss_fn)
    losses, metrics = trainer.train(train_loader, val_loader, n_epochs=n_epochs)

    with open(os.path.join(save_path, 'model_config.json'), 'w') as file:
        json.dump(config, file)

    with open(os.path.join(save_path, 'vectorizer.pkl'), 'wb') as file:
        pickle.dump(vectorizer, file)

    torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
