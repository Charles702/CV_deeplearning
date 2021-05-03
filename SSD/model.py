import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F


def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.
    
    # calcuate indices of boxes which contain object
    ann_confidence = ann_confidence.view(-1, 4)
    ann_box = ann_box.view(-1,4)
    pred_confidence = pred_confidence.view(-1,4)
    pred_box = pred_box.view(-1,4)


    # sum confidence expect the "background" dimension
    # convert one-hot -> class-id
    class_indices = torch.where(ann_confidence == 1)[1]
    #print("size of class - ",len(class_indices))
    obj_indices = torch.where(class_indices<3)[0]
    noobj_indices = torch.where(class_indices == 3)[0]
    # cells contain object
    obj_conf = pred_confidence[obj_indices]
    obj_conf_target = class_indices[obj_indices]
    obj_box = pred_box[obj_indices]
    obj_target = ann_box[obj_indices]
    # cells don't contain object
    noobj_conf = pred_confidence[noobj_indices]
    noobj_conf_target = class_indices[noobj_indices]   

    loss_obj = F.cross_entropy(obj_conf, obj_conf_target)
    loss_noobj = F.cross_entropy( noobj_conf, noobj_conf_target)
    loss_cls = loss_obj + 3*loss_noobj
    loss_boxes = F.smooth_l1_loss(obj_box, obj_target)

    loss = loss_cls + loss_boxes

    return loss
        




class conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding = 0):
        super(conv, self).__init__()
        self.layer = nn.Sequential(
             nn.Conv2d(in_ch, out_ch, kernel_size, stride = stride, padding = padding),
             nn.BatchNorm2d(out_ch),
             nn.ReLU(inplace=True)
          )

    def forward(self, x):
        x = self.layer(x)
        return x

class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        self.part1 = nn.Sequential(
            conv(3,64,3,2,1),   # 320 - 160
            conv(64,64,3,1,1),  #160 -160
            conv(64,64,3,1,1), # 160 - 160
            conv(64,128,3,2,1), # 160 -80
            conv(128,128,3,1,1), # 80 -80
            conv(128,128,3,1,1), # 80 -80
            conv(128,256,3,2,1), # 80 -40
            conv(256,256,3,1,1), # 40 -40
            conv(256,256,3,1,1), # 40 -40
            conv(256,512,3,2,1), # 40 -20
            conv(512,512,3,1,1), # 20 -20
            conv(512,512,3,1,1), # 20 -20
            conv(512,256,3,2,1), # 20 -10 
        )
        #main branch
        self.branch_m1 = conv(256,256,1,1)
        self.branch_m2 = conv(256,256,3,2,1) #10- 5

        self.branch_m3 = conv(256,256,1,1) # 5-5
        self.branch_m4 = conv(256,256,3,1) # 5-3

        self.branch_m5 = conv(256,256,1,1) # 3-3
        self.branch_m6 = conv(256,256,3,1) # 3-1

        self.branch_m_box = nn.Conv2d(256,16,1,1) #256 - 16
        self.branch_m_conf = nn.Conv2d(256,16,1,1) # 256 -16

        #side_branch
        self.branch2_box_100 = nn.Conv2d(256,16,3,1,1) # 10-10
        self.branch2_conf_100 = nn.Conv2d(256,16,3,1,1) # 10 - 10

        self.branch2_box_25 = nn.Conv2d(256,16,3,1,1) # 5 - 5
        self.branch2_conf_25 = nn.Conv2d(256,16,3,1,1)  # 5-5

        self.branch2_box_9 = nn.Conv2d(256,16,3,1,1) # # 3 -3
        self.branch2_conf_9 = nn.Conv2d(256,16,3,1,1) # 3-3  
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        #TODO: define forward
        x = self.part1(x)
        branch_100 = x

        x = self.branch_m1(x)
        x = self.branch_m2(x)
        branch_25 =x

        x = self.branch_m3(x)
        x = self.branch_m4(x)
        branch_9 =x

        x = self.branch_m5(x)
        x = self.branch_m6(x)
        branch_box_1 = self.branch_m_box(x)
        branch_conf_1 = self.branch_m_conf(x)
        # output1
        branch_box_1 = branch_box_1.reshape(-1,16,1)
        branch_conf_1 = branch_conf_1.reshape(-1,16,1)
        
        #ouput2 100
        branch_box_100 = self.branch2_box_100(branch_100)
        #print("branch_box_100 --",branch_box_100.size())
        branch_box_100 = branch_box_100.view(-1,16,100)
        branch_conf_100 = self.branch2_conf_100(branch_100)
        #print("branch_conf_100 --",branch_conf_100.size())
        branch_conf_100 = branch_conf_100.view(-1,16,100)

        #output3 25
        branch_box_25 = self.branch2_box_25(branch_25)
        branch_conf_25 = self.branch2_conf_25(branch_25)
        branch_box_25 = branch_box_25.view(-1,16,25)
        branch_conf_25 = branch_conf_25.view(-1,16,25)

        #output4 9
        branch_box_9 = self.branch2_box_9(branch_9)
        branch_conf_9 = self.branch2_conf_9(branch_9)
        branch_box_9 = branch_box_9.view(-1,16,9)
        branch_conf_9 = branch_conf_9.view(-1,16,9)

        #output bboxes
        bboxes = torch.cat((branch_box_100,branch_box_25,branch_box_9,branch_box_1), 2)
        bboxes = bboxes.permute(0,2,1)
        #print("bboxes--should be (?, 135,16)--",bboxes.size())  
        bboxes = bboxes.reshape(-1,540,self.class_num)

        #output confidence
        confidence = torch.cat((branch_conf_100,branch_conf_25,branch_conf_9,branch_conf_1), 2)      
        confidence = confidence.permute(0,2,1)
        #print("confidenc --should be(?,135,16)--", confidence.size())
        confidence = confidence.reshape(-1,540, self.class_num)
        
        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        
        return confidence,bboxes










