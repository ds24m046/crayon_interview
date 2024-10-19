import torch


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, X, y, vectorizer, encoder):
        
        self.vectorizer = vectorizer
        self.encoder = encoder
        
        self.X = self.vectorizer.transform(X)
        self.y = self.encoder.transform(y.reshape(-1, 1))
    
    def __len__(self):
    
        return self.X.shape[0]

    def __getitem__(self, index):
        
        X = torch.tensor(self.X[index].todense(), dtype=torch.float32, requires_grad=True)
        y = torch.tensor(self.y[index].todense(), dtype=torch.float32, requires_grad=False)
        
        if torch.cuda.is_available():
            X = X.to('cuda')
            y = y.to('cuda')
        
        return X, y
