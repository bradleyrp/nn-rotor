from torch import nn

from ConvBlock import ConvBlock, ConvTransposedBlock


class ConvDecoderStride(nn.Module):
    def __init__(
        self,
        n,
        out_channels,
        hidden_channels,
        kernel
    ):
        
        super().__init__()
            
        layers = []
            
        for i in range(n):
            
            in_ch = hidden_channels
            
            if i == n - 1:
                out_ch = out_channels
                act = "σ"
            else:
                out_ch = hidden_channels
                act = "relu"
            
            layers += [ConvTransposedBlock(in_ch, out_ch, kernel=kernel, act=act, stride=2)]
                    
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


class ConvDecoderPool(nn.Module):
    def __init__(
        self,
        n,
        out_channels,
        hidden_channels,
        kernel
    ):
        
        super().__init__()
            
        layers = []
            
        for i in range(n):
            
            in_ch = hidden_channels
            
            if i == n - 1:
                out_ch = out_channels
                act = "σ"
            else:
                out_ch = hidden_channels
                act = "relu"
            
            layers += [ConvBlock(in_ch, out_ch, kernel=kernel, act=act)]
            
            layers += [nn.Upsample(scale_factor=2)]
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
