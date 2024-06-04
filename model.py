import torch
import torch.nn as nn
import torch.nn.functional as F

#img_size is a tuple in format (H, W)
class MLP(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.Linear(3*img_size[0]*img_size[1], 2048),
            nn.ELU(),
            nn.Linear(2048, img_size[0]*img_size[1]),
        )
    def forward(self, x):
        x = torch.flatten(x, start_dim = 1)
        x = self.net(x)
        x = torch.unflatten(x,1,(1,self.img_size[0],self.img_size[1]))
        return F.sigmoid(x)

class ConvNet(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 5, padding = 2),
            nn.ELU(),
            nn.Conv2d(128, 128, 5, padding = 2),
            nn.ELU(),
            nn.Conv2d(128, 64, 3, padding = 1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.ELU(),
            nn.Conv2d(64, 16, 3,padding = 1),
            nn.ELU(),
            nn.Conv2d(16, 16, 3,padding = 1),
            nn.ELU(),
            nn.Conv2d(16, 4, 3,padding = 1),
            nn.ELU(),
            nn.Conv2d(4, 1, 3,padding = 1),
        )
    def forward(self, x):
        x = self.net(x)
        return F.sigmoid(x)
class ConvNet_A(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 7, padding = 3),
            nn.ELU(),
            nn.Conv2d(128, 128, 7, padding = 3),
            nn.ELU(),
            nn.Conv2d(128, 128, 7,padding = 3),
            nn.ELU(),
            nn.Conv2d(128, 64, 7,padding = 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 5,padding = 2),
            nn.ELU(),
            nn.Conv2d(64, 16, 5,padding = 2),
            nn.ELU(),
            nn.Conv2d(16, 16, 3,padding = 1),
            nn.ELU(),
            nn.Conv2d(16, 4, 3,padding = 1),
            nn.ELU(),
            nn.Conv2d(4, 1, 3,padding = 1),
        )
    def forward(self, x):
        x = self.net(x)
        return F.sigmoid(x)
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, residual = True):
        super().__init__()
        self.residual = residual
        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size , padding = kernel_size//2),
            nn.BatchNorm2d(out_channel),
            nn.ELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size , padding = kernel_size//2),
            nn.BatchNorm2d(out_channel),
        )
        self.shortcut = nn.Sequential()
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size = 1),
                nn.BatchNorm2d(out_channel),
            )
    def forward(self, x):
        if self.residual:
            return F.elu(self.main(x)+self.shortcut(x))
        return F.elu(self.main(x))

class ResBlock_Downsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3 , stride = 2,  padding = 1),
            nn.BatchNorm2d(out_channel),
            nn.ELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size , padding = kernel_size//2),
            nn.BatchNorm2d(out_channel),
        )
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size = 1),
                nn.AvgPool2d(2),
                nn.BatchNorm2d(out_channel),
            )
        else:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2),
                nn.BatchNorm2d(out_channel),
            )
    def forward(self, x):
        return F.elu(self.main(x)+self.shortcut(x))
        
class ResBlock_Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, residual = True):
        super().__init__()
        self.residual = True
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channel , out_channel, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size , padding = kernel_size//2),
            nn.BatchNorm2d(out_channel),
        )
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size = 1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.BatchNorm2d(out_channel),
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.BatchNorm2d(out_channel),
            )
    def forward(self, x):
        if self.residual:
            return F.elu(self.main(x)+self.shortcut(x))
        return F.elu(self.main(x))
    
class ResNet(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 7, padding = 3),#
            nn.MaxPool2d(kernel_size = 2),#
            nn.ELU(),#
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock(128, 64, 5),
            ResBlock(64, 16, 3),
            ResBlock_Upsample(16, 16, 3),
            nn.Conv2d(16, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ELU(),
            nn.Conv2d(4, 1, 3, padding = 1),
        )
    def forward(self, x):
        x = self.net(x)
        return F.sigmoid(x)
        
class ResNet_B(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 7, padding = 3),#
            nn.MaxPool2d(kernel_size = 2),#
            nn.ELU(),#
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock(128, 64, 5),
            ResBlock(64, 16, 3),
            nn.ConvTranspose2d(16,16,3, stride=2, padding=1, output_padding=1),#
            nn.BatchNorm2d(16),#
            nn.ELU(),#
            nn.Conv2d(16, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ELU(),
            nn.Conv2d(4, 1, 3, padding = 1),
        
        )
    def forward(self, x):
        x = self.net(x)
        return F.sigmoid(x)
class ResNet_BB(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 7, padding = 3),#
            nn.MaxPool2d(kernel_size = 2),#
            nn.ELU(),#
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 5),
            ResBlock(128, 128, 5),
            ResBlock(128, 128, 5),
            ResBlock(128, 128, 5),
            ResBlock(128, 128, 5),
            ResBlock(128, 128, 5),
            ResBlock(128, 64, 5),
            ResBlock(64, 16, 3),
            nn.ConvTranspose2d(16,16,3, stride=2, padding=1, output_padding=1),#
            nn.BatchNorm2d(16),#
            nn.ELU(),#
            nn.Conv2d(16, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ELU(),
            nn.Conv2d(4, 1, 3, padding = 1),
        
        )
    def forward(self, x):
        x = self.net(x)
        return F.sigmoid(x)
        
class ResNet_I(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 7, padding = 3),#
            nn.MaxPool2d(kernel_size = 2),#
            nn.ELU(),#
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock_Downsample(128, 256, 5),
            ResBlock(256, 256, 5),
            ResBlock(256, 128, 3),
            nn.ConvTranspose2d(128,64,3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64,16,3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ELU(),
            nn.Conv2d(4, 1, 3, padding = 1),
        )
    def forward(self, x):
        x = self.net(x)
        return F.sigmoid(x)


class MyModule(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.ecd_1 = nn.Sequential(
            nn.Conv2d(3, 128, 7, padding = 3),#
            nn.MaxPool2d(kernel_size = 2),#
            nn.ELU(),#
            ResBlock(128, 128, 5),
            ResBlock(128, 128, 5),
            ResBlock_Downsample(128, 128, 5),
            ResBlock(128, 128, 3),
            ResBlock(128, 128, 3),
            ResBlock_Downsample(128, 128, 3),
            ResBlock(128, 128, 3),
            ResBlock(128, 128, 3),
            ResBlock_Downsample(128, 128, 3),
            ResBlock(128, 128, 3),
            ResBlock(128, 64, 3),
        )
        self.ecd_2 = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, 32),
            nn.Tanh(),
        )
        self.dcd = nn.Sequential(
            ResBlock(35, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock(128, 128, 7),
            ResBlock(128, 64, 5),
            ResBlock(64, 16, 3),
            nn.Conv2d(16, 4, 3, padding = 1),
            nn.BatchNorm2d(4),
            nn.ELU(),
            nn.Conv2d(4, 1, 3, padding = 1),
        )
    def forward(self, x):
        latent = torch.flatten(self.ecd_1(x), start_dim = 1)
        latent = self.ecd_2(latent)
        latent = latent.unsqueeze(2).unsqueeze(3)
        latent = latent.repeat(1,1,self.img_size[0], self.img_size[1])
        x = torch.cat((x, latent), 1)
        x = self.dcd(x)
        return F.sigmoid(x)

class UNet(nn.Module):
    def __init__(self, img_size, residual = True):
        super().__init__()
        self.img_size = img_size
        self.residual = residual
        self.ecd1 = nn.Sequential(
            ResBlock(3, 64, 3, residual = residual),
            ResBlock(64, 64, 3, residual = residual),
        )
        self.ecd2 = nn.Sequential(
            ResBlock(64, 64, 3, residual = residual),
            ResBlock(64, 128, 3, residual = residual)
        )
        self.ecd3 = ResBlock(128, 256, 3, residual = residual)
        self.ecd4 = ResBlock(256, 512, 3, residual = residual)

        self.btnk = ResBlock(512, 1024, 3, residual = residual)

        self.up4 = ResBlock_Upsample(1024, 512, 3, residual = residual)
        self.dcd4 = ResBlock(1024, 512, 3, residual = residual)
        self.up3 = ResBlock_Upsample(512, 256, 3, residual = residual)
        self.dcd3 = ResBlock(512, 256, 3, residual = residual)
        self.up2 = ResBlock_Upsample(256, 128, 3, residual = residual)
        self.dcd2 = ResBlock(256, 128, 3, residual = residual)
        self.up1 = ResBlock_Upsample(128, 64, 3, residual = residual)
        self.dcd1 = ResBlock(128, 64, 3, residual = residual)

        self.output = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 1, 3, padding = 1),
        )

    def forward(self, x):
        enc1 = self.ecd1(x)
        enc2 = self.ecd2(F.max_pool2d(enc1, 2))
        enc3 = self.ecd3(F.max_pool2d(enc2, 2))
        enc4 = self.ecd4(F.max_pool2d(enc3, 2))

        btnk = self.btnk(F.max_pool2d(enc4, 2))

        dec4 = self.up4(btnk)
        dec4 = self.dcd4(torch.cat( (dec4, enc4) , dim=1))
        dec3 = self.up3(dec4)
        dec3 = self.dcd3(torch.cat( (dec3, enc3) , dim=1))
        dec2 = self.up2(dec3)
        dec2 = self.dcd2(torch.cat( (dec2, enc2) , dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dcd1(torch.cat( (dec1, enc1) , dim=1))

        output = self.output(dec1)
        return F.sigmoid(output)
        
