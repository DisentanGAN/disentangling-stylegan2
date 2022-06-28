import torch
from classifier import ResNet1D


if __name__ == "__main__":
    # TEST
    x = torch.randn((32, 128)) # reshaped into (batch_size, dim, 1) due to implementation
    print(len(x.shape))
    net = ResNet1D(in_channels=128, # (latent) dimension of input vector
                   base_filters=64,
                   kernel_size=3,
                   stride=1,
                   groups=1,
                   n_block=1,
                   n_classes=2,
                   #use_bn=False, # must only be True when batch_size > 1
                   verbose=True # debug option, prints shapes between layers
                   )
    net(x)
