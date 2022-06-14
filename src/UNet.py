""" Full assembly of the parts to form the complete network """

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, hidden_channels=64, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        ch_in, ch_out = n_channels, hidden_channels
        self.inc = DoubleConv(ch_in, ch_out)

        ch_in, ch_out = ch_out, ch_out * 2
        self.down1 = Down(ch_in, ch_out)

        ch_in, ch_out = ch_out, ch_out * 2
        self.down2 = Down(ch_in, ch_out)

        ch_in, ch_out = ch_out, ch_out * 2
        self.down3 = Down(ch_in, ch_out)

        factor = 2 if bilinear else 1
        
        ch_in, ch_out = ch_out, ch_out * 2
        self.down4 = Down(ch_in, ch_out // factor)

        ch_in, ch_out = ch_out, ch_out // 2
        self.up1 = Up(ch_in, ch_out // factor, bilinear)

        ch_in, ch_out = ch_out, ch_out // 2
        self.up2 = Up(ch_in, ch_out // factor, bilinear)

        ch_in, ch_out = ch_out, ch_out // 2
        self.up3 = Up(ch_in, ch_out // factor, bilinear)

        ch_in, ch_out = ch_out, ch_out // 2
        self.up4 = Up(ch_in, ch_out // factor, bilinear)

        assert ch_out == hidden_channels
        self.outc = OutConv(hidden_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
