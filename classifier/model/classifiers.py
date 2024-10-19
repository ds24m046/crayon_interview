import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class NNClassifier(torch.nn.Module):
    
    def __init__(self, n_input, n_target):
        
        super().__init__()

        self.n_input = n_input
        self.n_target = n_target
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_input, out_features=int(n_input / 2)),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=int(n_input / 2), out_features=int(n_input / 4)),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=int(n_input / 4), out_features=n_target)
        )
        
        self._init_weights(self.encoder)
        
    def _init_weights(self, module):
        
        for layer in module:
            
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)
    
    def forward(self, inputs):
        
        out = self.encoder(inputs)
        
        return out
