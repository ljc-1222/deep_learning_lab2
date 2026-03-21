import torch
from torch import nn

class UNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        
        self.conv_block_1 = nn.Sequential(
            
            nn.Conv2d(in_channels = 1, 
                      out_channels = 64, 
                      kernel_size = (3, 3)),   
               
            nn.ReLU(),
            
            nn.Conv2d(in_channels = 64, 
                      out_channels = 64, 
                      kernel_size = (3, 3)),  
            
            nn.ReLU()     
        )
        
        self.conv_block_2 = nn.Sequential(

            nn.Conv2d(in_channels = 64, 
                      out_channels = 128, 
                      kernel_size = (3, 3)),
            
            nn.ReLU(),
            
            nn.Conv2d(in_channels = 128, 
                      out_channels = 128, 
                      kernel_size = (3, 3)),    
            
            nn.ReLU()   
        )

        self.conv_block_3 = nn.Sequential(

            nn.Conv2d(in_channels = 128, 
                      out_channels = 256, 
                      kernel_size = (3, 3)),
            
            nn.ReLU(),
            
            nn.Conv2d(in_channels = 256, 
                      out_channels = 256, 
                      kernel_size = (3, 3)),      
            
            nn.ReLU() 
        )
        
        self.conv_block_4 = nn.Sequential(

            nn.Conv2d(in_channels = 256, 
                      out_channels = 512, 
                      kernel_size = (3, 3)),
            
            nn.ReLU(),
            
            nn.Conv2d(in_channels = 512, 
                      out_channels = 512, 
                      kernel_size = (3, 3)),     
            
            nn.ReLU() 
        )

        self.conv_block_5 = nn.Sequential(

            nn.Conv2d(in_channels = 512, 
                      out_channels = 1024, 
                      kernel_size = (3, 3)),
            
            nn.ReLU(),
            
            nn.Conv2d(in_channels = 1024, 
                      out_channels = 1024, 
                      kernel_size = (3, 3)),      
            
            nn.ReLU()
        )
        
        self.conv_block_6 = nn.Sequential(

            nn.Conv2d(in_channels = 1024,
                      out_channels = 512,
                      kernel_size = (3, 3)),
            
            nn.ReLU(),
            
            nn.Conv2d(in_channels = 512,
                      out_channels = 512,
                      kernel_size = (3, 3)),    
            
            nn.ReLU()         
        )        
        
        self.conv_block_7 = nn.Sequential(
            
            nn.Conv2d(in_channels = 512,
                      out_channels = 256,
                      kernel_size = (3, 3)),
            
            nn.ReLU(),

            nn.Conv2d(in_channels = 256,
                      out_channels = 256,
                      kernel_size = (3, 3)),            
            
            nn.ReLU() 
        )         
        
        self.conv_block_8 = nn.Sequential(

            nn.Conv2d(in_channels = 256,
                      out_channels =  128,
                      kernel_size = (3, 3)),
            
            nn.ReLU(),

            nn.Conv2d(in_channels = 128,
                      out_channels = 128,
                      kernel_size = (3, 3)),    
            
            nn.ReLU()         
        )        
        
        self.conv_block_9 = nn.Sequential(

            nn.Conv2d(in_channels = 128,
                      out_channels = 64,
                      kernel_size = (3, 3)),
            
            nn.ReLU(),

            nn.Conv2d(in_channels = 64,
                      out_channels = 64,
                      kernel_size = (3, 3)),    
            
            nn.ReLU(),

            nn.Conv2d(in_channels = 64,
                      out_channels = 2,
                      kernel_size = (1, 1)),
        )
        
        self.up_conv_block_1 = nn.ConvTranspose2d(in_channels = 1024,
                                                  out_channels = 512,
                                                  kernel_size = (2, 2),
                                                  stride = 2)
        
        self.up_conv_block_2 = nn.ConvTranspose2d(in_channels = 512,
                                                  out_channels = 256,
                                                  kernel_size = (2, 2),
                                                  stride = 2)
        
        self.up_conv_block_3 = nn.ConvTranspose2d(in_channels = 256,
                                                  out_channels = 128,
                                                  kernel_size = (2, 2),
                                                  stride = 2)
        
        self.up_conv_block_4 = nn.ConvTranspose2d(in_channels = 128,
                                                  out_channels = 64,
                                                  kernel_size = (2, 2),
                                                  stride = 2)
        
        self.max_pool_block = nn.MaxPool2d(kernel_size = (2, 2))


    def center_crop(self, src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _, _, h, w = src.shape
        _, _, th, tw = target.shape
        dh = (h - th) // 2
        dw = (w - tw) // 2
        return src[:, :, dh:dh + th, dw:dw + tw]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        enc1 = self.conv_block_1(x)
        enc2 = self.conv_block_2(self.max_pool_block(enc1))
        enc3 = self.conv_block_3(self.max_pool_block(enc2))
        enc4 = self.conv_block_4(self.max_pool_block(enc3))
        enc5 = self.conv_block_5(self.max_pool_block(enc4))
        
        up5 = self.up_conv_block_1(enc5)
        skip4 = self.center_crop(enc4, up5)
        up4 = self.up_conv_block_2(self.conv_block_6(torch.cat([skip4, up5], dim = 1)))
        skip3 = self.center_crop(enc3, up4)
        up3 = self.up_conv_block_3(self.conv_block_7(torch.cat([skip3, up4], dim = 1)))
        skip2 = self.center_crop(enc2, up3)
        up2 = self.up_conv_block_4(self.conv_block_8(torch.cat([skip2, up3], dim = 1)))
        skip1 = self.center_crop(enc1, up2)
        up1 = self.conv_block_9(torch.cat([skip1, up2], dim = 1))     
        
        return up1
    
if __name__ == "__main__":
    
    import torchinfo
    
    model = UNet()
    torchinfo.summary(model, input_size = (1, 1, 528, 528))
    