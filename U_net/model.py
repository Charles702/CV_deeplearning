import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.con1 = nn.Conv2d(1, 64, 3)
        self.bn1 = nn.BatchNorm2d(64) 
        self.con2 = nn.Conv2d(64, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.down1 = downStep(64,128)
        self.down2 = downStep(128,256)
        self.down3 = downStep(256,512)
        self.down4 = downStep(512,1024)
        
        self.up1 = upStep(1024,512)
        self.up2 = upStep(512,256)
        self.up3 = upStep(256,128)
        self.up4 = upStep(128,64, False)
        
        self.out = nn.Conv2d(64, n_classes, 1)
        
    def forward(self, x):
        x = self.con1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.con2(x)
        x = self.bn2(x)
        x = F.relu(x)      
        down_out1 = x
        
        x = self.down1(x)
        down_out2 = x
        
        x = self.down2(x)
        down_out3 = x
        x = self.down3(x)
        down_out4 = x
        
        x = self.down4(x)
        
        x = self.up1(x, down_out4)
        x = self.up2(x, down_out3)
        x = self.up3(x, down_out2)
        x = self.up4(x, down_out1)
        
        x = self.out(x)      
        
        return x

class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        self.down = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(inC, outC,3)
        self.bn1 = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(outC, outC, 3)
        self.bn2 = nn.BatchNorm2d(outC)
        
    def forward(self, x):
        x = self.down(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        return x

class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        # Do not forget to concatenate with respective step in contracting path
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.withReLu = withReLU
        
        self.up = nn.ConvTranspose2d(inC, outC, 2, stride = 2)  
        self.conv1 = nn.Conv2d(inC, outC, 3)
        self.bn1 = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(outC, outC, 3)
        self.bn2 = nn.BatchNorm2d(outC)
        
        
    def forward(self, x, x_down):
        x = self.up(x)
        
        #x combine x_down: crop x_down as same size of x
        size_x = x.size()[2]
        size_xdown = x_down.size()[2]
        offset = (size_xdown - size_x)//2
        x_crop = x_down[:,:, offset:offset+size_x, offset:offset+size_x]
        x = torch.cat((x, x_crop), 1) #concatenate along channel
        
        x = self.conv1(x)
        x = self.bn1(x)
        if self.withReLu:
            x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        if self.withReLu:
            x = F.relu(x)
          
        return x