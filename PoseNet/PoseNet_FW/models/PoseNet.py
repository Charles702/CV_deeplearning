import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle


def init(key, module, weights=None):
    if weights == None:
        return module

    # initialize bias.data: layer_name_1 in weights
    # initialize  weight.data: layer_name_0 in weights
    module.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
    module.weight.data = torch.from_numpy(weights[(key + "_0").encode()])

    return module


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, key=None, weights=None):
        super(InceptionBlock, self).__init__()

        # TODO: Implement InceptionBlock
        # Use 'key' to load the pretrained weights for the correct InceptionBlock
        
        key_n1x1 = 'inception_{}/1x1'.format(key)
        key_n3x3red = 'inception_{}/3x3_reduce'.format(key)
        key_n3x3 = 'inception_{}/3x3'.format(key)
        key_n5x5red = 'inception_{}/5x5_reduce'.format(key)
        key_n5x5 = 'inception_{}/5x5'.format(key)
        key_pool_proj = 'inception_{}/pool_proj'.format(key)        
        
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            init(key_n1x1, nn.Conv2d(in_channels, n1x1, kernel_size = 1), weights),
            nn.ReLU(True)
        )

        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(
           init(key_n3x3red, nn.Conv2d(in_channels, n3x3red, kernel_size = 1), weights),
           nn.ReLU(True),
           init(key_n3x3, nn.Conv2d(n3x3red, n3x3, kernel_size = 3, padding = 1), weights),
           nn.ReLU(True)
        )

        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(
           init(key_n5x5red, nn.Conv2d(in_channels, n5x5red, kernel_size = 1), weights),
           nn.ReLU(True),
           init(key_n5x5, nn.Conv2d(n5x5red, n5x5, kernel_size = 5, padding = 2), weights),
           nn.ReLU(True)
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding = 1),
            nn.ReLU(True),
            init(key_pool_proj, nn.Conv2d(in_channels, pool_planes, kernel_size = 1), weights),
            nn.ReLU(True)

        )

    def forward(self, x):
        # TODO: Feed data through branches and concatenate
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)

        return torch.cat((b1,b2,b3,b4), 1 )


class LossHeader(nn.Module):
    def __init__(self, key, weights=None):
        super(LossHeader, self).__init__()
        # TODO: Define loss headers
        if key == 'loss1':
            inchannel = 512
        else:    # loss2 
            inchannel = 528
        
        self.loss_P1 = nn.Sequential(
            nn.AvgPool2d(kernel_size= 5, stride= 3), #  check size ?? what is  V in graph
            init(key+'/conv', nn.Conv2d(inchannel, 128, kernel_size=1, stride=1), weights),
            nn.ReLU(True),
            nn.Flatten(),       
            #init(key+'/fc', nn.Linear(2048, 1024), weights),       
            nn.Linear(2048, 1024),
            nn.Dropout(0.7)
        )

        self.fc1 = nn.Linear(1024, 3)
        self.fc2 = nn.Linear(1024, 4)

    def forward(self, x):
        # TODO: Feed data through loss headers
        #print(x.size())
        x = self.loss_P1(x)

        xyz = self.fc1(x)
        wpqr = self.fc2(x)

        return xyz, wpqr


class PoseNet(nn.Module):
    def __init__(self, load_weights=True):
        super(PoseNet, self).__init__()

        # Load pretrained weights file
        if load_weights:
            print("Loading pretrained InceptionV1 weights...")
            file = open('pretrained_models/places-googlenet.pickle', "rb")
            weights = pickle.load(file, encoding="bytes")
            file.close()
        # Ignore pretrained weights
        else:
            weights = None

        # TODO: Define PoseNet layers

        self.pre_layers = nn.Sequential(
            # Example for defining a layer and initializing it with pretrained weights
            init('conv1/7x7_s2', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(True),
            
            nn.MaxPool2d(3, stride = 2, padding = 1),
            nn.ReLU(True),

            nn.LocalResponseNorm(size =5, alpha = 0.0001, beta = 0.75, k=1),
            init('conv2/3x3_reduce', nn.Conv2d(64, 64, kernel_size = 1, stride =1), weights),
            nn.ReLU(True),

            init('conv2/3x3', nn.Conv2d(64, 192, kernel_size =3, stride = 1, padding = 1), weights),
            nn.ReLU(True),

            nn.LocalResponseNorm(size =5, alpha = 0.0001, beta = 0.75, k=1),
            nn.MaxPool2d(3, stride =2, padding = 1),
            nn.ReLU(True)

        )

        # Example for InceptionBlock initialization
        # block inception3
        self.inc_3 = nn.Sequential(
            InceptionBlock(192, 64, 96, 128, 16, 32, 32, "3a", weights),
            InceptionBlock(256, 128, 128, 192, 32, 96, 64, "3b", weights))
            #InceptionBlock(256, 128, 128, 192, 32, 96, 64, "3c", weights))

        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        # block inception 4
        self.inc_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64, "4a", weights)
        self.inc_4middle  =  nn.Sequential(
            InceptionBlock(512, 160, 112, 224, 24, 64, 64, "4b", weights),
            InceptionBlock(512, 128, 128, 256, 24, 64, 64, "4c", weights),
            InceptionBlock(512, 112, 144, 288, 32, 64, 64, "4d", weights))
        self.inc_4e =  InceptionBlock(528, 256, 160, 320, 32, 128, 128, "4e", weights)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1 )

        # inception 5
        self.inc_5  =  nn.Sequential(             
            InceptionBlock(832, 256, 160, 320, 32, 128, 128, "5a", weights),
            InceptionBlock(832, 384, 192, 384, 48, 128, 128, "5b", weights))

        self.loss1 = LossHeader('loss1', weights)
        self.loss2 = LossHeader('loss2', weights)

        self.loss3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=7, stride=1),
            nn.Flatten(),
            nn.Linear(1024, 2048),
            nn.Dropout(0.4)
        )

        self.out1 = nn.Linear(2048, 3)
        self.out2 = nn.Linear(2048, 4)

        print("PoseNet model created!")

    def forward(self, x):
        # TODO: Implement PoseNet forward
        x = self.pre_layers(x)
        #print('pre-out', x.size())
        x = self.inc_3(x)
        #print('inc_3-out', x.size())
        x = self.maxpool1(x)
        #print('maxpool1-out' , x.size())
        
        x = self.inc_4a(x)
        #print('inc_4a-out' , x.size())
        loss1_xyz, loss1_wpqr = self.loss1(x)   
        #print(loss1_xyz)
        x = self.inc_4middle(x)
        #print('inc_4middle-out' , x.size())
        loss2_xyz, loss2_wpqr = self.loss2(x)      
        x = self.inc_4e(x)
        x = self.maxpool2(x)
        
        x = self.inc_5(x)
        x = self.loss3(x)
        loss3_xyz = self.out1(x)
        loss3_wpqr = self.out2(x)
           
        if self.training:
            return loss1_xyz, \
                   loss1_wpqr, \
                   loss2_xyz, \
                   loss2_wpqr, \
                   loss3_xyz, \
                   loss3_wpqr
        else:
            return loss3_xyz, \
                   loss3_wpqr



class PoseLoss(nn.Module):

    def __init__(self, w1_x, w2_x, w3_x, w1_q, w2_q, w3_q):
        super(PoseLoss, self).__init__()

        self.w1_x = w1_x
        self.w2_x = w2_x
        self.w3_x = w3_x
        self.w1_q = w1_q
        self.w2_q = w2_q
        self.w3_q = w3_q
        
        self.loss1_xyz = nn.MSELoss()
        self.loss1_wpqr = nn.MSELoss()
        self.loss2_xyz = nn.MSELoss()
        self.loss2_wpqr = nn.MSELoss()
        self.loss3_xyz = nn.MSELoss()
        self.loss3_wpqr = nn.MSELoss()


    def forward(self, p1_xyz, p1_wpqr, p2_xyz, p2_wpqr, p3_xyz, p3_wpqr, poseGT):
        # TODO: Implement loss  
        # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr
        #print(poseGT.size())  (75,3) (75,4)
        bath_size = p1_xyz.size()[0]
        gt_xyz = poseGT[:,0:3]
        gt_wpqr = poseGT[:,3:7]
        gt_norm_wpqr = torch.norm(gt_wpqr,dim = 1, keepdim = True )
        
        #loss1 = torch.norm(p1_xyz- gt_xyz, dim = 1, keepdim = True) + self.w1_q * torch.norm(p1_wpqr - gt_wpqr/gt_norm_wpqr, dim = 1, keepdim = True)
        #loss2 = torch.norm(p2_xyz- gt_xyz, dim = 1, keepdim = True) + self.w2_q * torch.norm(p2_wpqr - gt_wpqr/gt_norm_wpqr, dim = 1, keepdim = True)
        #loss3 = torch.norm(p3_xyz- gt_xyz, dim = 1, keepdim = True) + self.w3_q * torch.norm(p3_wpqr - gt_wpqr/gt_norm_wpqr, dim = 1, keepdim = True)
        #loss = torch.sum(self.w1_x* loss1 + self.w2_x*loss2 + self.w3_x*loss3)/bath_size
        

        loss1 = self.loss1_xyz(p1_xyz, gt_xyz) + self.w1_q * self.loss1_wpqr(p1_wpqr, gt_wpqr/gt_norm_wpqr)
        loss2 = self.loss2_xyz(p2_xyz, gt_xyz) + self.w2_q * self.loss2_wpqr(p2_wpqr, gt_wpqr/gt_norm_wpqr)
        loss3 = self.loss3_xyz(p3_xyz, gt_xyz) + self.w3_q * self.loss2_wpqr(p3_wpqr, gt_wpqr/gt_norm_wpqr)
        loss = self.w1_x* loss1 + self.w2_x*loss2 + self.w3_x*loss3
       
        
        
        return loss
