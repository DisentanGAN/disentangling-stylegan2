import torch.nn as nn


class LinearClassifier(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.layer(x)
        x = nn.functional.sigmoid(x)
        return x


class NonLinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Linear(in_dim, 128)
        )

        for _ in range(3):
            self.layers.append(
                nn.Linear(128, 128)
            )

        self.classification = nn.Linear(128, out_dim)
        

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nn.functional.relu(x)

        x = self.classification(x)
        x = nn.functional.sigmoid(x)

        return x