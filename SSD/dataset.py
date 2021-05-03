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
import numpy as np
import os
import cv2

#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    size1,size2,size3,size4 = layers 
    num_cell = size1**2 +size2**2 +size3**2 + size4**2
    # #output:
    # #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    # #TODO:
    # #create an numpy array "boxes" to store default bounding boxes
    # #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    # #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    # #the second dimension 4 means each cell has 4 default bounding boxes.
    # #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    # #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    # #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    # #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    #boxes = np.zeros((num_cell,4,8))
    boxes = []
    for c in range(len(layers)):
      size_g = layers[c]
      ls = large_scale[c]
      ss = small_scale[c]
      box_shape = [[ss,ss],[ls,ls],[ls*np.sqrt(2),ls/np.sqrt(2)],[ls/np.sqrt(2), ls*np.sqrt(2)]]
      for i in range(1,size_g+1,1):
        for j in range(1,size_g+1,1):
          #print(i,j)
          center_x = i/size_g - (0.5)/size_g
          center_y = j/size_g - (0.5)/size_g
          boxes_cell = []
          for shape in box_shape:
            #print(shape)
            width, height = shape
            # correct x_min y_min, if they are out of boundary
            x_min = max(center_x - width/2,0)
            y_min = max(center_y - height/2,0)
            x_max = min(center_x + width/2,1)
            y_max = min(center_y + height/2,1)

            box = [center_x, center_y, width, height, x_min, y_min, x_max, y_max]
            #print(box)
            boxes_cell.append(box)
            #print(c , i , j)
          boxes.append(boxes_cell)
    boxes = np.array(boxes)
    #print("length of boxes", boxes.shape)
    boxes =np.reshape(boxes,((num_cell*4),8))
    #print("size of boxes", boxes.shape)
    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)

    #print("ious---", ious)
    
    ious_true = ious>threshold
    #TODO:
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #caulcuate ann_box [relative_center_x, relative_center_y, relative_width, relative_height]
    g_center_x = (x_max + x_min)/2
    g_center_y = (y_max + y_min)/2 
    g_w = x_max - x_min
    g_h = y_max - y_min

    indices = np.where(ious_true == 1)[0]
    #print("indices----",indices)
    for idx in indices:
      #print("idx", idx)
      #print("boxs_default[idx]--", boxs_default[idx])
      px, py, pw, ph = boxs_default[idx][0], boxs_default[idx][1],boxs_default[idx][2], boxs_default[idx][3]
      r_center_x = (g_center_x - px)/pw
      r_center_y = (g_center_y - py)/ph
      r_width = np.log(g_w/pw)
      r_height = np.log(g_h/ph)
      ann_box[idx] = [r_center_x, r_center_y, r_width, r_height]
      
      # reset hot vector
      #print("cat_id",cat_id)
      ann_confidence[idx,-1] = 0
      ann_confidence[idx, int(cat_id)] = 1
    
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    
    if len(indices) == 0:
      ious_true = np.argmax(ious)
      #print("max--", ious_true)
      px, py, pw, ph = boxs_default[ious_true][0], boxs_default[ious_true][1],boxs_default[ious_true][2], boxs_default[ious_true][3]
      r_center_x = (g_center_x - px)/pw
      r_center_y = (g_center_y - py)/ph
      r_width = np.log(g_w/pw)
      r_height = np.log(g_h/ph)
      ann_box[ious_true] = [r_center_x, r_center_y, r_width, r_height]

      
      # reset hot vector
      ann_confidence[ious_true,-1] = 0
      ann_confidence[ious_true,int(cat_id)] = 1 

    #TODO:
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)


from os import path

class COCO(torch.utils.data.Dataset): 
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        #print(sorted(os.listdir(self.imgdir)))
        self.img_names = sorted(os.listdir(self.imgdir))

        if anndir != "":  #training or validate
          train_size = round(len(self.img_names)*0.9)
          if self.train == True: # train
            self.img_names = self.img_names[:train_size][:5]
          else: 
            print("validate data loading....")
            self.img_names = self.img_names[train_size:]
        else:
          #self.img_names = self.img_names[:30]
          print("test data loading ...")

        self.image_size = image_size
        
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        
        #TODO:
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        #print("img_name--",img_name)
        # read image
        img = cv2.imread(img_name)
        img_w = img.shape[1]
        img_h = img.shape[0]
        # read annotation in file 
        ann_info = []
        x_min_total = img_w
        y_min_total = img_h
        x_max_total = 0
        y_max_total = 0

        #process annotation file
        if path.exists(ann_name):          # no annotation file for test
          ann_file = open(ann_name,"r")
          for data in ann_file:
              p = data.split(" ")
              cat_id, g_x, g_y, g_w, g_h  = float(p[0]), float(p[1]),float(p[2]),float(p[3]),float(p[4])
              x_min_g = g_x
              y_min_g = g_y
              x_max_g = g_x + g_w
              y_max_g = g_y + g_h
              # if multiply object, calculate the min of combination
              if x_min_g < x_min_total:
                x_min_total = x_min_g
              if y_min_g < y_min_total:
                y_min_total = y_min_g
              if x_max_g > x_max_total:
                x_max_total = x_max_g
              if y_max_g > y_max_total:
                y_max_total = y_max_g
              #store the min x y, max x,y respect to original size image
              ann_info.append([cat_id, x_min_g, y_min_g, x_max_g, y_max_g])    
        # else:
        #    print("Test data don't have annotation file")
        
        #  self.train ----False    don't apply Augmentation
        # data augmentation
        if self.train == True: 
            # calcuate the random area of cropped image
            # need to cover ground true boxes
            offset_l = np.random.random_sample()
            offset_r = np.random.random_sample()
            offset_t = np.random.random_sample()
            offset_b = np.random.random_sample()

            x_move =  x_min_total*offset_l
            x_s = x_move
            x_e = x_max_total + (img_w - x_max_total)*offset_r

            y_move =  y_min_total*offset_t
            y_s = y_move
            y_e = y_max_total + (img_h - y_max_total)*offset_b

            # random crop new image
            img = img[round(y_s):round(y_e), round(x_s): round(x_e)]
            # get new size of cropped image
            img_w = img.shape[1]
            img_h = img.shape[0]

            # relocate the position of gt boxes in new image
            for b in ann_info:
              #print("===anno==before aug==",b)
              cat_id = b[0]
              new_x_min = (b[1] - x_move)/img_w
              new_y_min = (b[2] - y_move)/img_h
              new_x_max = (b[3] - x_move)/img_w
              new_y_max = (b[4] - y_move)/img_h
              match(ann_box,ann_confidence, self.boxs_default,self.threshold, cat_id ,new_x_min , new_y_min, new_x_max, new_y_max)
        else: # evalucate or test
              for b in ann_info:
                #print(b)
                #convert ground truth size to [0,1]
                img_w = img.shape[1]
                img_h = img.shape[0]
                #calculate ground truth box
                cat_id = b[0]
                x_min_g = b[1]/img_w
                y_min_g = b[2]/img_h
                x_max_g = b[3]/img_w
                y_max_g = b[4]/img_h 
              #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
                match(ann_box,ann_confidence, self.boxs_default,self.threshold,cat_id,x_min_g, y_min_g, x_max_g, y_max_g)

        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        
        #to use function "match":
        #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        
        image = cv2.resize(img, (self.image_size,self.image_size))
        image = np.array(image, dtype= np.float32)
        # convert to shape (batch, 3, 320, 320)
        image = np.transpose(image,(2,0,1))
        #print("after transpose----", image.shape)
        # normalization
        image = torch.from_numpy(image/255.0)
        ann_box = torch.from_numpy(ann_box)
        ann_confidence = torch.from_numpy(ann_confidence)
        orignal_shape = np.array([img_w,img_h])
        
        return image, ann_box, ann_confidence, self.img_names[index], orignal_shape
