from torch import nn, flatten
from lib.neuralNetworks.residualUnit import residualStack

class encoder(nn.Module):
    """ Encoder neural network consisting of 4 convolutional layers. """
    def __init__(self, latent_dim):
        super().__init__()
        self.activation = nn.ReLU()
        self.pool   = nn.MaxPool1d(3)
        self.conv1  = nn.Conv1d(1, 16, 7)
        self.conv2  = nn.Conv1d(16, 32, 6)
        self.conv3  = nn.Conv1d(32, 64, 6)
        self.conv4  = nn.Conv1d(64, 128, 5)
        self.dense1 = nn.Linear(1280, latent_dim)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.3)
        
    def forward(self, input):
        out = self.pool(self.activation(self.conv1(input)))
        out = self.bn1(out)
        out = self.drop(out)
        out = self.pool(self.activation(self.conv2(out)))
        out = self.bn2(out)
        out = self.drop(out)
        out = self.pool(self.activation(self.conv3(out)))
        out = self.bn3(out)
        out = self.drop(out)
        out = self.pool(self.activation(self.conv4(out)))
        out = self.bn4(out)
        out = self.drop(out)
        out = flatten(out, 1)
        out = self.dense1(out)
        return out 



class residualEncoder(nn.Module):
    """ Encoder neural network consisting of 4 residual unis. """
    def __init__(self, latent_dim):
        super().__init__()
        self.pool   = nn.MaxPool1d(3)
        self.drop = nn.Dropout(0.3)
        self.resUnit1 = residualStack(1, 16, kernel_size=3)
        self.resUnit2 = residualStack(16, 32, kernel_size=3)
        self.resUnit3 = residualStack(32, 64, kernel_size=3)
        self.resUnit4 = residualStack(64, 128, kernel_size=3)
        self.dense    = nn.Linear(128*12, latent_dim)

    def forward(self, x):
        x = self.pool(self.resUnit1(x))       
        x = self.pool(self.resUnit2(x))
        x = self.pool(self.resUnit3(x))
        x = self.pool(self.resUnit4(x))
        x = flatten(x, 1)
        x = self.dense(x)
        return x