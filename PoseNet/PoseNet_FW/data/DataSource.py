import os
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
import numpy as np

class DataSource(data.Dataset):
    def __init__(self, root, resize=256, crop_size=224, train=True):
        self.root = os.path.expanduser(root)
        self.resize = resize
        self.crop_size = crop_size
        self.train = train

        self.image_poses = []
        self.images_path = []

        self._get_data()

        # TODO: Define preprocessing

        # Load mean image
        mean_image_path = os.path.join(self.root, 'mean_image.npy')
        if os.path.exists(mean_image_path):
            self.mean_image = np.load(mean_image_path)
            print("Mean image loaded!")
        else:
            self.mean_image = self.generate_mean_image()

    def _get_data(self):

        if self.train:
            txt_file = self.root + 'dataset_train.txt'
        else:
            txt_file = self.root + 'dataset_test.txt'

        with open(txt_file, 'r') as f:
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            for line in f:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                self.images_path.append(self.root + fname)

    def generate_mean_image(self):
        print("Computing mean image:")

        # TODO: Compute mean image
        
        # Initialize mean_image
        #t numpy stores images as (height, width)
        mean_arr =np.zeros((256,455,3),np.float)
        
        import time
        s = time.time()
        # Iterate over all training images
        for file_path in self.images_path:
            orig_img = Image.open(file_path)
            
            # PIL stores images as (width, height)
            width, height = orig_img.size
            # (1920, 1080)
            #print(orig_img.size)
            
            #calculate new height and width
            im_height = self.resize
            im_width = round(width/height *self.resize)  
            # height:256  width:455
            
            #resize image
            img = orig_img.resize((im_width, im_height))
            # sum the image
            mean_arr = mean_arr + np.array(img, dtype = np.float) 
            
        #get mean image
        mean_image = mean_arr/len(self.images_path)
        print("exectute time:" ,time.time()-s)
        print(mean_arr)
        
        # round to integer 
        #arr= numpy.array(numpy.round(mean_arr),dtype=numpy.uint8)
        #mean_img =Image.fromarray(arr,mode="RGB")     
        #mean_img.save("mean_image.npy")
        #mean_img.show()

        # Store mean image
        np.save(os.path.join(self.root, 'mean_image.npy'), mean_image)
        print("Mean image computed!")

        return mean_image

    def __getitem__(self, index):
        """
        return the data of one image
        """
        img_path = self.images_path[index]
        img_pose = self.image_poses[index]

        data = Image.open(img_path)

        # TODO: Perform preprocessing
        img_pil = data.resize((455 , 256))
        img_arr = np.array(img_pil, dtype = np.float) - self.mean_image
        
        train_tansfomer = T.Compose([
                T.ToTensor(),
                T.RandomCrop(self.crop_size),
                T.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5],)
            ])
        
        test_tansfomer = T.Compose([
                T.ToTensor(),
                T.CenterCrop(self.crop_size),
                T.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5],)
            ])        
        
        if self.train:
            input_data  = train_tansfomer(img_arr)
        else:
            input_data  = test_tansfomer(img_arr)
                      
        return input_data, img_pose

    def __len__(self):
        return len(self.images_path)