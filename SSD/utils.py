import numpy as np
import cv2
from dataset import *



colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    image_ = image_*255.0    #recover
    #print("image",image_)
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    img_with = image.shape[1]
    img_height = image.shape[0]

    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #calculate ground truth bounding (x1,y1)  (x2,y2)
                r_c_x, r_c_y, r_w, r_h = ann_box[i]
                #print("ann_box[i]",i,ann_box[i])
                default_b_x, default_b_y,defualt_b_w, default_b_h = boxs_default[i][0:4]   
                # recover grouth true boxes
                g_c_x = r_c_x*defualt_b_w + default_b_x
                g_c_y = r_c_y*default_b_h + default_b_y
                g_w = defualt_b_w*np.exp(r_w)
                g_h = default_b_h*np.exp(r_h)    
                x1 = g_c_x - g_w/2
                y1 = g_c_y - g_h/2
                x2 = g_c_x + g_w/2
                y2 = g_c_y + g_h/2
                start_point = ( round(x1*img_with) , round(y1*img_height))
                end_point = ( round(x2*img_with) , round(y2*img_height))        
                image1 = cv2.rectangle(image1, start_point, end_point, colors[j], 2)
                #print("gt===",colors[j], start_point , end_point)    


                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                d_x1, d_y1, d_x2, d_y2 = boxs_default[i][4:] 
                d_start_point = ( round(d_x1*img_with) , round(d_y1*img_height))
                d_end_point = ( round(d_x2*img_with) , round(d_y2*img_height))   
                #print("default box===",d_start_point, d_end_point)        
                image2 = cv2.rectangle(image2, d_start_point, d_end_point, colors[j], 2)
                
   

                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)
    
    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #print("preconf ", i, j, pred_confidence[i,j])
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                r_c_x, r_c_y, r_w, r_h = pred_box[i]
                #print("pred_box[i]",i,pred_box[i])
                default_b_x, default_b_y,defualt_b_w, default_b_h = boxs_default[i][0:4]   
                # recover actual position
                g_c_x = r_c_x*defualt_b_w + default_b_x
                g_c_y = r_c_y*default_b_h + default_b_y
                g_w = defualt_b_w*np.exp(r_w)
                g_h = default_b_h*np.exp(r_h)    
                x1 = g_c_x - g_w/2
                y1 = g_c_y - g_h/2
                x2 = g_c_x + g_w/2
                y2 = g_c_y + g_h/2
                start_point = ( round(x1*img_with) , round(y1*img_height))
                end_point = ( round(x2*img_with) , round(y2*img_height))               
                image3 = cv2.rectangle(image3, start_point, end_point, colors[j], 2)
                #print("pred===",colors[j], start_point , end_point)     
                # plt.imshow(image3)
                # plt.show()  

                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                d_x1, d_y1, d_x2, d_y2 = boxs_default[i][4:] 
                d_start_point = ( round(d_x1*img_with) , round(d_y1*img_height))
                d_end_point = ( round(d_x2*img_with) , round(d_y2*img_height))                               
                image4 = cv2.rectangle(image4, d_start_point, d_end_point, colors[j], 2)   

    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    cv2.imwrite( windowname +'.png', image)
    #cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    cv2.waitKey(1)
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.






def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.3, threshold=0.8):
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    
    #get the bounding box with highest confidence.
    select_boxes = []
    classes =[]
    actual_boxes = np.zeros((540,8))
    conf_c = np.copy(confidence_)
   
    # pre_box is relative value,  convert relative box -> actual boxes
    # [r_center_x, r_center_y, r_w, r_h]
    Px = boxs_default[:,0]
    Py = boxs_default[:,1]
    Pw = boxs_default[:,2]
    Ph = boxs_default[:,3]

    actual_boxes[:,0] = box_[:,0]*Pw + Px
    actual_boxes[:,1] = box_[:,1]*Ph + Py
    actual_boxes[:,2] = Pw*np.exp(box_[:,2])
    actual_boxes[:,3] = Ph*np.exp(box_[:,3])
    actual_boxes[:,4] = actual_boxes[:,0] - actual_boxes[:,2]/2   # min_x
    actual_boxes[:,5] = actual_boxes[:,1] - actual_boxes[:,3]/2   # min_y
    actual_boxes[:,6] = actual_boxes[:,0] + actual_boxes[:,2]/2   # min_x
    actual_boxes[:,7] = actual_boxes[:,1] + actual_boxes[:,3]/2   # min_y

    selected_idx_box = []
    # find the indices of object confidenc > 0.5 
    max_confidence_matrix = np.max(conf_c[:,0:3], axis=1)
    indices_boxes_obj = np.where(max_confidence_matrix > threshold)[0]
    #print("indices_boxes_obj  ", indices_boxes_obj)
    #print(" confidenc maxtrix  ", max_confidence_matrix[indices_boxes_obj])

    if len(indices_boxes_obj) == 0:
        #print("No box confidence is higher than threshlod, assign it the biggist one")
        top = np.argmax(max_confidence_matrix)
        selected_idx_box.append(top)

    while len(indices_boxes_obj) > 0: 
        k = np.argmax(max_confidence_matrix[indices_boxes_obj])
        
        # get idx of highest confidence
        highest_idx = indices_boxes_obj[k]
        #print("HHH---", highest_idx, max_confidence_matrix[highest_idx])
        selected_idx_box.append(highest_idx)
        # remove the highest from dict
        indices_boxes_obj = np.delete(indices_boxes_obj,k)
        # box in backup list
        actual_boxes_bak = actual_boxes[indices_boxes_obj]
        # 
        highest_box = actual_boxes[highest_idx]
        # compare rest box in dictionary with highest , calculate the iou
        ious = iou(actual_boxes_bak, highest_box[4],highest_box[5],highest_box[6],highest_box[7])
        #print("ious--",ious)
        #print(" find the overlap boxes: " , np.where(np.array(ious)>overlap)[0])       
        overlap_list = np.where(np.array(ious)>overlap)[0] 
        indices_boxes_obj = np.delete(indices_boxes_obj, overlap_list)
        #print("indices_boxes_obj --after nms ", indices_boxes_obj) 

    # the box will be showed
    #print(selected_idx_box)

    # boxes not in "selected_idx_box" confidence will be set to "ground"
    # for c in range(len(confidence_)):
    #   if c not in (selected_idx_box):
    #     confidence_[c] = [0,0,0,0.99]

    #print(confidence_[selected_idx_box])
    #print(box_[selected_idx_box])
  
    return selected_idx_box
    #TODO: non maximum suppression


def visualize_pred_nms(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default, selected_idx_box):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    image_ = image_*255.0    #recover
    #print("image",image_)
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    img_with = image.shape[1]
    img_height = image.shape[0]

    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #calculate ground truth bounding (x1,y1)  (x2,y2)
                r_c_x, r_c_y, r_w, r_h = ann_box[i]
                #print("ann_box[i]",i,ann_box[i])
                default_b_x, default_b_y,defualt_b_w, default_b_h = boxs_default[i][0:4]   
                # recover grouth true boxes
                g_c_x = r_c_x*defualt_b_w + default_b_x
                g_c_y = r_c_y*default_b_h + default_b_y
                g_w = defualt_b_w*np.exp(r_w)
                g_h = default_b_h*np.exp(r_h)    
                x1 = g_c_x - g_w/2
                y1 = g_c_y - g_h/2
                x2 = g_c_x + g_w/2
                y2 = g_c_y + g_h/2
                start_point = ( round(x1*img_with) , round(y1*img_height))
                end_point = ( round(x2*img_with) , round(y2*img_height))        
                image1 = cv2.rectangle(image1, start_point, end_point, colors[j], 2)
                #print("gt===",colors[j], start_point , end_point)    


                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                d_x1, d_y1, d_x2, d_y2 = boxs_default[i][4:] 
                d_start_point = ( round(d_x1*img_with) , round(d_y1*img_height))
                d_end_point = ( round(d_x2*img_with) , round(d_y2*img_height))   
                #print("default box===",d_start_point, d_end_point)        
                image2 = cv2.rectangle(image2, d_start_point, d_end_point, colors[j], 2)
                
   

                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)
    
    #pred
    #print("============", selected_idx_box)
    for select_id in selected_idx_box:
          #print("preconf ",  pred_confidence[select_id])
          color_id = np.argmax(pred_confidence[select_id])

          #TODO:
          #image3: draw network-predicted bounding boxes on image3
          r_c_x, r_c_y, r_w, r_h = pred_box[select_id]
          #print("pred_box[i]",i,pred_box[i])
          default_b_x, default_b_y,defualt_b_w, default_b_h = boxs_default[select_id][0:4]   
          # recover actual position
          g_c_x = r_c_x*defualt_b_w + default_b_x
          g_c_y = r_c_y*default_b_h + default_b_y
          g_w = defualt_b_w*np.exp(r_w)
          g_h = default_b_h*np.exp(r_h)    
          x1 = g_c_x - g_w/2
          y1 = g_c_y - g_h/2
          x2 = g_c_x + g_w/2
          y2 = g_c_y + g_h/2
          start_point = ( round(x1*img_with) , round(y1*img_height))
          end_point = ( round(x2*img_with) , round(y2*img_height))               
          image3 = cv2.rectangle(image3, start_point, end_point, colors[color_id], 2)
          #print("pred===",colors[j], start_point , end_point)     
          # plt.imshow(image3)
          # plt.show()  

          #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
          d_x1, d_y1, d_x2, d_y2 = boxs_default[i][4:] 
          d_start_point = ( round(d_x1*img_with) , round(d_y1*img_height))
          d_end_point = ( round(d_x2*img_with) , round(d_y2*img_height))                               
          image4 = cv2.rectangle(image4, d_start_point, d_end_point, colors[color_id], 2)   

    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    cv2.imwrite(windowname +'.png' , image)
    #cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    cv2.waitKey(1)
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.



#--- save text

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes : cat dog person
def output_test(windowname, pred_confidence, pred_box, image_, boxs_default, selected_idx_box, orignal_shape):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    test_ann_root = 'output/test/annotations/'
  
    img_with = orignal_shape[0]
    img_height = orignal_shape[1]

    for select_id in selected_idx_box:
        #print("preconf ",  pred_confidence[select_id])
        color_id = np.argmax(pred_confidence[select_id])
        if color_id == 3:  # didn't predict any object, so propose the background
            color = (255, 255, 255)
        else: 
            color = colors[color_id]
            
        r_c_x, r_c_y, r_w, r_h = pred_box[select_id]
        #print("pred_box[i]",i,pred_box[i])
        default_b_x, default_b_y,defualt_b_w, default_b_h = boxs_default[select_id][0:4]   
        # recover actual position
        g_c_x = r_c_x*defualt_b_w + default_b_x
        g_c_y = r_c_y*default_b_h + default_b_y
        g_w = defualt_b_w*np.exp(r_w)
        g_h = default_b_h*np.exp(r_h)    
        x1 = g_c_x - g_w/2
        y1 = g_c_y - g_h/2
        x2 = g_c_x + g_w/2
        y2 = g_c_y + g_h/2
        start_point = ( round(x1*img_with) , round(y1*img_height))
        end_point = ( round(x2*img_with) , round(y2*img_height))                      
        #print("pred===",colors[j], start_point , end_point)     
        # plt.imshow(image3)
        # plt.show()  
        #genrete annotation texts: (cat_id, x, y, w, h)
        content = '%d %f %f %f %f\n' %(color_id, x1*img_with, y1*img_height, g_w*img_with, g_h*img_height)
        fn = test_ann_root + windowname[:-4] +'.txt'
        with open(fn,'a') as ann_file:
          ann_file.write(content)









