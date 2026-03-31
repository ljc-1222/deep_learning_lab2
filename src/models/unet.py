import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3),
            nn.ReLU(inplace = True),
        )
        self._initialize_weights() 

    def _initialize_weights(self):
        for m in self.block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.up4  = nn.ConvTranspose2d(1024, 512, kernel_size = 2, stride = 2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)
        self.dec3 = DoubleConv(512, 256)

        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
        self.dec2 = DoubleConv(256, 128)

        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
        self.dec1 = DoubleConv(128, 64)

        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.head = nn.Conv2d(64, 1, kernel_size = 1)

    @staticmethod
    def center_crop(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        _, _, h,  w  = src.shape
        _, _, th, tw = target.shape
        dh, dw = (h - th) // 2, (w - tw) // 2
        return src[:, :, dh:dh + th, dw:dw + tw]

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        # Decoder path: upsample → crop-and-concat skip → double conv
        u4 = self.up4(b);  d4 = self.dec4(torch.cat([self.center_crop(e4, u4), u4], dim = 1))
        u3 = self.up3(d4); d3 = self.dec3(torch.cat([self.center_crop(e3, u3), u3], dim = 1))
        u2 = self.up2(d3); d2 = self.dec2(torch.cat([self.center_crop(e2, u2), u2], dim = 1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([self.center_crop(e1, u1), u1], dim = 1))

        return self.head(d1)


if __name__ == "__main__":
    
    import torchinfo

    model = UNet()
    torchinfo.summary(model, input_size = (1, 3, 364, 364))
