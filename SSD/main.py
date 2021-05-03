import argparse
import os
import numpy as np
import time
import cv2

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

from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test


class_num = 4 #cat dog person background

num_epochs = 150
batch_size = 16


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])

#Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True


if not args.test:
    print("Start training...")
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, image_size=320)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = optim.Adam(network.parameters(), lr = 1e-4)
    #feel free to try other optimizers and parameters.
    
    print("training data set :", len(dataset))
    start_time = time.time()

    for epoch in range(num_epochs):
        #TRAINING
        network.train()

        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_, _, _= data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1

        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        
        # SoftMax conver to probability
        pred_confidence = F.softmax(pred_confidence, 2)
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()

        viaul_name1 = 'train_%d'%(epoch)
        viaul_name2 = 'NMS_train_%d'%(epoch)
        x = 0
        pred_confidence_ = pred_confidence[x].detach().cpu().numpy()
        pred_box_ = pred_box[x].detach().cpu().numpy()
        # before NMS
        visualize_pred(viaul_name1, pred_confidence_, pred_box_, ann_confidence_[x].numpy(), ann_box_[x].numpy(), images_[x].numpy(), boxs_default)
        # After NMS
        selected_idx_box = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        visualize_pred_nms(viaul_name2, pred_confidence_, pred_box_, ann_confidence_[x].numpy(), ann_box_[x].numpy(), images_[x].numpy(), boxs_default, selected_idx_box)


        #VALIDATION
        network.eval()
        
        # TODO: split the dataset into 90% training and 10% validation
        # use the training set to train and the validation set to evaluate
        
        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_, _, _= data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_conf_val, pred_box_val = network(images)
            #-----softmax
            pred_conf_val = F.softmax(pred_conf_val, 2)
            
            pred_conf_val_ = pred_conf_val.detach().cpu().numpy()
            pred_box_val_ = pred_box_val.detach().cpu().numpy()
            
            #optional: implement a function to accumulate precision and recall to compute mAP or F1.
            #update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
        
        #visualize
        x=0
        viaul_name1 = 'val_%d'%(epoch)
        viaul_name2 = 'NMS_val_%d'%(epoch)
        visualize_pred(viaul_name1, pred_conf_val_[x], pred_box_val_[x], ann_confidence_[x].numpy(), ann_box_[x].numpy(), images_[x].numpy(), boxs_default)

        selected_idx_box = non_maximum_suppression(pred_conf_val_[x],pred_box_val_[x],boxs_default)
        visualize_pred_nms(viaul_name2, pred_conf_val_[x], pred_box_val_[x], ann_confidence_[x].numpy(), ann_box_[x].numpy(), images_[x].numpy(), boxs_default, selected_idx_box)

        #optional: compute F1
        #F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        #print(F1score)
        
        #save weights
        if epoch%10==9:
            #save last network
            print('saving net...')
            torch.save(network.state_dict(), 'output/network_%d.pth' %(epoch))
else:
    print("this is test")
    #TEST
    dataset_test = COCO("data/test/images/", "", class_num, boxs_default, train = False, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('output/network_149.pth'))
    network.eval()
    

    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_, image_names_ , orignal_shape = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()
    
        pred_conf_test, pred_box_test = network(images)
    
        #-----softmax
        pred_conf_test = F.softmax(pred_conf_test, 2)   
        pred_conf_test_ = pred_conf_test.detach().cpu().numpy()
        pred_box_test_ = pred_box_test.detach().cpu().numpy()
    
        # [Important] Disable data augmentation when generating those bounding box!! 
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment
        
        
        # pred_confidence_ = pred_confidence[x].detach().cpu().numpy()
        # pred_box_ = pred_box[x].detach().cpu().numpy()
        for i in range(len(pred_conf_test_)):
            x= i
            #Visualize compare before and after NMS
            selected_idx_box = non_maximum_suppression(pred_conf_test_[x], pred_box_test_[x],boxs_default)
            # generate images and texts
            output_test(image_names_[x], pred_conf_test_[x], pred_box_test_[x], images_[x].numpy(), boxs_default, selected_idx_box, orignal_shape[x].numpy())

