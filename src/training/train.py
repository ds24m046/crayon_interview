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
input_path = 'inputs'
save_path = 'artifacts'

if __name__ == "__main__":

    with open(os.path.join(save_path, 'data.json'), 'r') as file:
        data = json.load(file)

    with open(os.path.join(save_path, 'tag2idx.json'), 'r') as file:
        tag2idx = json.load(file)
    
    print('Vectorizing dataset...')
    # TODO: logging

    vectorizer = TfidfVectorizer(min_df=5, max_df=0.75, analyzer='word', stop_words='english')
    vectorizer.fit(data['X_train'])
    
    X_train = vectorizer.transform(data['X_train'])
    X_test = vectorizer.transform(data['X_test'])
    
    distribution = np.asarray(X_train.sum(axis=0)).squeeze()
    distribution = distribution / X_train.sum()
    
    encoder = OneHotEncoder()
    encoder.fit(np.array(data['y_train']).reshape(-1, 1))
    
    y_train = encoder.transform(np.array(data['y_train']).reshape(-1, 1))
    y_test = encoder.transform(np.array(data['y_test']).reshape(-1, 1))
    
    train_dataset = datasets.Dataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = datasets.Dataset(X_test, y_test)
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

    with open(os.path.join(save_path, 'distribution.pkl'), 'wb') as file:
        pickle.dump(distribution, file)
    
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
    
    os.remove(os.path.join(input_path, 'data.csv'))