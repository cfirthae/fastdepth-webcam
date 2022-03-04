import numpy as np
import requests
import jetson.inference
import jetson.utils
from segnet_utils import *

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True



from torchvision import datasets, transforms
from PIL import Image
import cv2
import time

net = jetson.inference.segNet("fcn-resnet18-sun-640x512") # load segNet
filename = input("Filename:")
# set Gstreamer pipeline - regular cv2.VideoCapture(0) doesnt work for RPi v2

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=15,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def main():

	pth = torch.load('best_result.pth.tar') # load fastdepth model w Torch
	model = pth['model'] # index correct model path

	image = Image.open(filename) #Image.open('image.jpg') # loads PIL image from captured frame   
	image = image.resize((224,224),Image.ANTIALIAS) # resize to 224x224 with AA filtering
	    
	img_resize = np.array(image) # convert PIL to np array

	transform = transforms.Compose([transforms.ToTensor()]) 
	depth_img = transform(image) # uses above function to make resized image into pytorch tensor
	    
	    ### Segmentation Section ###
	seg_img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
	seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2RGBA).astype(np.float32) # color conversions to correct segmentation input


	seg_img = jetson.utils.cudaFromNumpy(seg_img) # convert from np array to cuda

	net.Process(seg_img) # process img in model

	net.Overlay(seg_img)
	jetson.utils.cudaDeviceSynchronize()

	img_np = jetson.utils.cudaToNumpy(seg_img)

	img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR).astype(np.float32)
	img_cv2 = img_np.copy()
	cv2.imwrite(filename + '_seg.jpg', img_cv2)

	class_mask = jetson.utils.cudaAllocMapped(width=224, height=224, format='gray8')
	net.Mask(class_mask,224,224) # create and assign mask array of class IDs

	class_mask_np = jetson.utils.cudaToNumpy(class_mask) # cuda to np array

	class_blacklist = (0,1,8,9,2,13,15) # class ID blacklist 
	class_mask = np.reshape(class_mask_np, [224,224]) # elimininates extra dimension


	    ### Depth Map Section ###
	x = depth_img.resize(1,3,224,224)
	x_torch = x.type(torch.cuda.FloatTensor)


	depth = model(x_torch) #returns torch.Tensor of shape torch.Size([1,1,224,224])
	#the above line takes the longest to run and is the result of the first frame wait time

	depth_min = depth.min()
	depth_max = depth.max()
	max_val = (2**(8))-1 # 255

	if depth_max - depth_min > np.finfo("float").eps: # min != max?
	       out = max_val * (depth - depth_min) / (depth_max - depth_min)
		#returns torch.Tensor of shape torch.Size([1,1,224,224])
	else:
		out = np.zeros(depth.shape, dtype=depth.type)

	out = out.cpu().detach().numpy()
	out = out.reshape(224,224)  
	    
	out = Image.fromarray(out) # creates PIL Image obj from above array
	out = out.convert('L')  # converts image to grayscale 
	    
	out = np.array(out)
	cv2.imwrite(filename + '_depth.jpg', out)


	outFiltered = np.where((np.isin(class_mask,class_blacklist)),255,out)


	#find max val index using below 
	    
	outMin = np.where(outFiltered == np.amin(outFiltered))
	#print(out_min)

	cv2.imwrite(filename + '_filtered.jpg', outFiltered)		
	out[outMin[0],outMin[1]] = 255
	cv2.imshow('Depth Map Output', out)

			

	cv2.destroyAllWindows()

	return



if __name__ == '__main__': 
    main()
