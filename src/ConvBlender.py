from torch import nn


class ConvBlender(nn.Module):
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel
    ):
        
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
        
        self.layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding="same")
        
    def forward(self, x):
        return self.layer(x)
