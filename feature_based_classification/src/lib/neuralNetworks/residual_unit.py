from torch import nn


class ResidualBlock(nn.Module):
    """ Residual Block, consisting of two channel and size invariant convolutions with skip connection. """
    def __init__(self, channel_num, kernel_size, mode='encode'):
        super().__init__()

        padding = int((kernel_size-1)/2)
        if mode=='encode':
            conv1 = nn.Conv1d(channel_num, channel_num, kernel_size, padding=padding)
            conv2 = nn.Conv1d(channel_num, channel_num, kernel_size, padding=padding)
        elif mode=='decode':
            conv1 = nn.ConvTranspose1d(channel_num, channel_num, kernel_size, padding=padding)
            conv2 = nn.ConvTranspose1d(channel_num, channel_num, kernel_size, padding=padding)
        
        self.conv_block1 = nn.Sequential(conv1, nn.BatchNorm1d(channel_num), nn.ReLU()) 
        self.conv_block2 = nn.Sequential(conv2, nn.BatchNorm1d(channel_num), nn.ReLU()) 
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.activation(x)
        return out



class ResidualStack(nn.Module):
    """ Residual Unit consisting of multiple residual Blocks. """
    def __init__(self, channel_in, channel_out, kernel_size, mode='encode'):
        super().__init__()

        # if necessary, perform dimension transformation to output channel
        if channel_in == channel_out:
            self.dimension_mismatch = False
        else:
            self.dimension_mismatch = True
            if mode=='encode':
                self.dim_transition = nn.Conv1d(channel_in, channel_out, kernel_size=1)
            elif mode=='decode':
                self.dim_transition = nn.ConvTranspose1d(channel_in, channel_out, kernel_size=1)

        self.res_blk1 = ResidualBlock(channel_out, kernel_size, mode)
        self.res_blk2 = ResidualBlock(channel_out, kernel_size, mode)

    def forward(self, x):
        x = self.dim_transition(x) if self.dimension_mismatch else x
        x = self.res_blk1(x)
        x = self.res_blk2(x)
        return(x)