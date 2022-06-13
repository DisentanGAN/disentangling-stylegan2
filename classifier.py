import torch.nn as nn

class LinearClassifier(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer = nn.Linear(in_features=in_dim, out_features=out_dim)
        
    def forward(self, x):
        x = self.layer(x)
        x = nn.functional.softmax(x, dim=1)
        return x