import tqdm
import torch
import numpy as np
from . import evaluators


class Trainer:
    
    def __init__(self, model, optimizer, loss):
        
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
    
    def _train_epoch(self, loader):
        
        self.model.train()
        
        losses = []
        progress = tqdm.tqdm(loader)
        for n, (X_batch, y_batch) in enumerate(progress):
            
            self.optimizer.zero_grad()
            
            logits = self.model(X_batch).squeeze()
            target = y_batch.squeeze()
            
            loss = self.loss(logits, target)
            
            loss.backward()
            self.optimizer.step()
            
            losses.append(float(loss))
            progress.set_postfix(loss='{:<.06f}'.format(np.mean(losses)))
        
        return np.mean(losses)
    
    def _validate_epoch(self, loader):
                
        self.model.eval()
        
        y_true = []
        y_pred = []
        progress = tqdm.tqdm(loader)
        
        for n, (X_batch, y_batch) in enumerate(progress):
            
            with torch.no_grad():
                logits = self.model(X_batch).squeeze()
                probas = torch.sigmoid(logits).numpy()
            
            y_batch = y_batch.squeeze().numpy()
            
            y_true += list(y_batch.argmax(axis=1))
            y_pred += list(probas.argmax(axis=1))
        
        metric = evaluators.evaluate_model(y_true, y_pred)
        print('\n' + str(metric) + '\n')
        
        return metric
        
    def train(self, train_loader, val_loader, n_epochs=1):
        
        stats = {
            'loss': [],
            'metrics': []
        }
        
        for epoch in range(n_epochs):
            
            loss = self._train_epoch(train_loader)
            metric = self._validate_epoch(val_loader)
            
            stats['loss'] += [loss]
            stats['metrics'] += [metric]
            
        return stats    
        