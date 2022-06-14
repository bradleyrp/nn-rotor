from torch import nn

from ConvBlock import ConvBlock

class ConvEncoderStride(nn.Module):
    def __init__(
        self,
        n,
        in_channels,
        hidden_channels,
        kernel
    ):
        
        super().__init__()
            
        layers = []
            
        for i in range(n):
            
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = hidden_channels
            
            layers += [ConvBlock(in_ch, out_ch, kernel, stride=2)]
                    
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


class ConvEncoderPool(nn.Module):
    def __init__(
        self,
        n,
        in_channels,
        hidden_channels,
        kernel
    ):
        
        super().__init__()
            
        layers = []
            
        for i in range(n):
            
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = hidden_channels
            
            layers += [ConvBlock(in_ch, out_ch, kernel)]
            
            layers += [nn.MaxPool1d(2)]
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
