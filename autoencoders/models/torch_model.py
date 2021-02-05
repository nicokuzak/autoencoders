import torch.nn as nn
import torch.nn.functional as F


class TorchModel(nn.Module):
    """An instance of class TorchModel is an AutoEncoder with 4 encoding layers and 4 decoding layers. 
    
    The layers themselves are pretty well defined, but depending on data preprocessing and user preferences, 
    the user can vary the input dimension and the final encoding dimension.
    
    Args:
        input_dim (int): Dimension of the input (most likely 29)
        encoding_dim (int): Dimension of the final encoding vector. This is the lowest dimensional vector and should be < 12.
    """
    def __init__(self, input_dim, filter_sizes):
        """Initialize the AutoEncoder for the Credit Card Data."""
        super(TorchModel, self).__init__()  # Inherit nn.Module
        self.enc1 = nn.Linear(input_dim, filter_sizes[0])
        self.enc2 = nn.Linear(filter_sizes[0], filter_sizes[1])
        self.enc3 = nn.Linear(filter_sizes[1], filter_sizes[2])
        self.enc4 = nn.Linear(filter_sizes[2], filter_sizes[3])

        self.dec1 = nn.Linear(filter_sizes[3], filter_sizes[2])
        self.dec2 = nn.Linear(filter_sizes[2], filter_sizes[1])
        self.dec3 = nn.Linear(filter_sizes[1], filter_sizes[0])
        self.dec4 = nn.Linear(filter_sizes[0], input_dim)

    def forward(self, x):
        """Defines one forward pass through the autoencoder. We use tanh as to have an output between -1 and 1, which is what our data was mostly scaled to.
        
        Args:
            x (numpy.ndarray): A batch of inputs to the model."""
        x = F.tanh(self.enc1(x))
        x = F.tanh(self.enc2(x))
        x = F.tanh(self.enc3(x))
        x = F.tanh(self.enc4(x))

        x = F.tanh(self.dec1(x))
        x = F.tanh(self.dec2(x))
        x = F.tanh(self.dec3(x))
        x = self.dec4(x)
        return x