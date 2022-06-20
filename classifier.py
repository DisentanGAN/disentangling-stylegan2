import torch.nn as nn


class LinearClassifier(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.layer(x)
        x = nn.functional.softmax(x, dim=1)
        return x


# class AdaptiveClassifier(nn.Module):
#     def __init__(self, in_dim, out_dim, channels):
#         super().__init__()
#         self.layers = nn.ModuleList()
        
#         self.layers.append(
#             nn.Linear(in_dim, channels[0]))
#         for i in range(len(channels) - 1):
#             self.layers.append(
#                 nn.Linear(channels[i], out_features=channels[i+1]))

#         self.classification = nn.Linear(channels[-1], out_dim)

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#             x = nn.functional.relu(x)
        
#         x = self.classification(x)
#         x = nn.functional.softmax(x, dim=1)

#         return x