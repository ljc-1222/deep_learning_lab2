import torch
from torch import nn


# ── Shared building blocks ─────────────────────────────────────────────────────

class BasicBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels))
            
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + self.shortcut(x))


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _make_layer(in_channels: int, out_channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:

    layers: list[nn.Module] = [BasicBlock(in_channels, out_channels, stride = stride)]
    
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_channels, out_channels))
        
    return nn.Sequential(*layers)


# ── Main model ─────────────────────────────────────────────────────────────────

class ResNet34UNet(nn.Module):

    def __init__(self, in_channels: int = 3, num_classes: int = 2) -> None:
        super().__init__()

        # ── Encoder: ResNet-34 ─────────────────────────────────────────────────
        self.stem = nn.Sequential(                                      
            nn.Conv2d(in_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = _make_layer(64,  64,  num_blocks = 3)             
        self.layer2 = _make_layer(64,  128, num_blocks = 4, stride = 2) 
        self.layer3 = _make_layer(128, 256, num_blocks = 6, stride = 2) 
        self.layer4 = _make_layer(256, 512, num_blocks = 3, stride = 2)

        # ── Decoder: UNet-style ────────────────────────────────────────────────
        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)
        self.dec4 = DoubleConv(512, 256)  

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
        self.dec3 = DoubleConv(256, 128)    

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
        self.dec2 = DoubleConv(128, 64)  

        self.up1  = nn.ConvTranspose2d(64, 64, kernel_size = 2, stride = 2)
        self.dec1 = DoubleConv(128, 64)     

        self.up0  = nn.ConvTranspose2d(64, 32, kernel_size = 2, stride = 2)
        self.head = nn.Conv2d(32, num_classes, kernel_size = 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Encoder path
        s  = self.stem(x)               
        e1 = self.layer1(self.maxpool(s)) 
        e2 = self.layer2(e1)             
        e3 = self.layer3(e2)             
        b  = self.layer4(e3)             

        # Decoder path: upsample → concat skip → double conv
        u4 = self.up4(b);  d4 = self.dec4(torch.cat([e3, u4], dim = 1))
        u3 = self.up3(d4); d3 = self.dec3(torch.cat([e2, u3], dim = 1))
        u2 = self.up2(d3); d2 = self.dec2(torch.cat([e1, u2], dim = 1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([s,  u1], dim = 1))
        u0 = self.up0(d1)

        return self.head(u0)


if __name__ == "__main__":
    import torchinfo

    model = ResNet34UNet()
    torchinfo.summary(model, input_size = (1, 3, 512, 512))
