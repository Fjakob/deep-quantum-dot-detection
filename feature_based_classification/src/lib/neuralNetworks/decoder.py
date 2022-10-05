from torch import nn, sigmoid


class Decoder(nn.Module):
    """ Decoder neural network consisting of 4 inverted convolutional layers. """
    def __init__(self, latent_dim):
        super().__init__()
        self.activation = nn.ReLU()
        self.dense1   = nn.Linear(latent_dim, 128*36)
        self.conv1T   = nn.ConvTranspose1d(128, 64, 7, stride=3)
        self.conv2T   = nn.ConvTranspose1d(64, 32, 6, stride=3)
        self.conv3T   = nn.ConvTranspose1d(32, 16, 6, stride=3)
        self.conv4T   = nn.ConvTranspose1d(16, 1, 5)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(16)
            
    def forward(self, input):
        out = self.activation(self.dense1(input))
        out = out.view(-1, 128, 36)
        out = self.bn1(out)
        out = self.activation(self.conv1T(out))
        out = self.bn2(out)
        out = self.activation(self.conv2T(out))
        out = self.bn3(out)
        out = self.activation(self.conv3T(out))
        out = self.bn4(out)
        out = self.conv4T(out)
        return out  



class DeepDecoder(nn.Module):
    """ Decoder neural network consisting of 6 inverted convolutional layers. """
    def __init__(self, latent_dim):
        super().__init__()
        self.activation = nn.ReLU()
        self.dense1   = nn.Linear(latent_dim, 128*36)
        self.conv1T   = nn.ConvTranspose1d(128, 64, 7, stride=3)
        self.conv2T   = nn.ConvTranspose1d(64, 32, 6, stride=3)
        self.conv3T   = nn.ConvTranspose1d(32, 16, 6, stride=3)
        self.conv4T   = nn.ConvTranspose1d(16, 8, 3, padding=1)
        self.conv5T   = nn.ConvTranspose1d(8, 4, 3, padding=1)
        self.conv6T   = nn.ConvTranspose1d(4, 1, 5)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(16)
        self.bn5 = nn.BatchNorm1d(8)
        self.bn6 = nn.BatchNorm1d(4)
            
    def forward(self, input):
        out = self.activation(self.dense1(input))
        out = out.view(-1, 128, 36)
        out = self.bn1(out)
        out = self.activation(self.conv1T(out))
        out = self.bn2(out)
        out = self.activation(self.conv2T(out))
        out = self.bn3(out)
        out = self.activation(self.conv3T(out))
        out = self.bn4(out)
        out = self.activation(self.conv4T(out))
        out = self.bn5(out)
        out = self.activation(self.conv5T(out))
        out = self.bn6(out)
        out = sigmoid(self.conv6T(out))
        return out