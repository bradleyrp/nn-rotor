from torch import nn

from utils import find_activation


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel,
                 stride=1, act="relu", padding="same"
                ):
        
        super().__init__()
        
        layer_act = find_activation(act)
        
        if padding == "same":
            padding = (kernel - 1) // 2
                
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            layer_act
        ]
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)
    
    
class ConvTransposedBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel,
                 stride=1, act="relu", padding="same"
                ):
        
        super().__init__()
                    
        layer_act = find_activation(act)
        
        if stride == 2:
            output_padding = 1
        else:
            output_padding = 0
            
        if padding == "same":
            padding = (kernel - 1) // 2
                
        layers = [
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            ),
            nn.BatchNorm1d(out_channels),
            layer_act
        ]
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)
