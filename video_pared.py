import os
import csv
import numpy as np

import jetson.inference
import jetson.utils
from segnet_utils import *

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

import models
from metrics import AverageMeter, Result
import utils



from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time
'''
from torchviz import make_dot
import onnx
import onnxruntime as ort
'''


net = jetson.inference.segNet("fcn-resnet18-sun-640x512") # load segNet

# set Gstreamer pipeline - regular cv2.VideoCapture(0) doesnt work for RPi v2

def mask_image(image, mask, masked_value):
    
    scale_factor = image.shape[0]//mask.shape[0] # how much the mask needs to be scaled up by to match the image's size
    
    resized_mask = mask.repeat(scale_factor,axis=0).repeat(scale_factor,axis=1)

    return(np.where((resized_mask==0),masked_value,image)) # where the mask==0, return the masked value, else return the value of the original array.

def seg_filter( a, blacklist ):
    s = a.shape
    l = np.ndarray.flatten( a )
    ol = [ e not in blacklist for e in l ]
    return np.reshape( ol, s ).astype(int)

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
	#print(model)
	pth = torch.load('best_result.pth.tar')
	model = pth['model'] 
	#model.eval()


	#cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
	class_mask = None
	cap = cv2.VideoCapture('full_hallway_color.avi')
	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	#width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	#height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	#fps = cap.get(cv2.CAP_PROP_FPS)

	out_depth = cv2.VideoWriter('depth_out.avi',fourcc,10,(224,224),False)
	while cap.isOpened():		
		start = time.time()
		ret, frame = cap.read()


		image = Image.fromarray(frame) #Image.open('image.jpg') # loads PIL image from captured frame
		    
		image = image.resize((224,224),Image.ANTIALIAS) # resize to 224x224 with AA filtering
		    
		img_resize = np.array(image)
		    #cv2.imshow('image after resize',cv2_img)
		transform = transforms.Compose([transforms.ToTensor()]) 
		depth_img = transform(image) # uses above function to make resized image into pytorch tensor
		    
		    ### Segmentation Section ###
		seg_img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
		seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2RGBA).astype(np.float32)


		seg_img = jetson.utils.cudaFromNumpy(seg_img)

		net.Process(seg_img)
		    
		grid_width, grid_height = net.GetGridSize()

		class_mask = jetson.utils.cudaAllocMapped(width=224, height=224, format='gray8')
		net.Mask(class_mask,224,224)

		class_mask_np = jetson.utils.cudaToNumpy(class_mask)

		class_blacklist = (0,1,2,13)
		class_mask = np.reshape(class_mask_np, [224,224])

		net.Overlay(seg_img)
		jetson.utils.cudaDeviceSynchronize()

		seg_img = jetson.utils.cudaToNumpy(seg_img)

		seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGBA2BGR).astype(np.float32)

		#cv2.imwrite("seg.jpeg",seg_img)  
		cv2.imshow('Segmentation Output', seg_img/255)           

		    ### Depth Map Section ###
		x = depth_img.resize(1,3,224,224)
		    #x = torch.rand(1,3,224,224)
		x_torch = x.type(torch.cuda.FloatTensor)
		   
		depth = model(x_torch) #returns torch.Tensor of shape torch.Size([1,1,224,224])

		depth_min = depth.min()
		depth_max = depth.max()
		max_val = (2**(8))-1 # 255

		if depth_max - depth_min > np.finfo("float").eps:
		       out = max_val * (depth - depth_min) / (depth_max - depth_min)
			#returns torch.Tensor of shape torch.Size([1,1,224,224])
		else:
			out = np.zeros(depth.shape, dtype=depth.type)

		out = out.cpu().detach().numpy()  
		out = out.reshape(224,224)  
		    
		out = Image.fromarray(out) # creates PIL Image obj from above array
		out = out.convert('L')  # converts image to grayscale 
		    
		out = np.array(out)

		out_filtered = np.where((np.isin(class_mask,class_blacklist)),255,out)


		    #find max val index using below 
		    
		out_min = np.where(out_filtered == np.amin(out))
		out_filtered[out_min[0],out_min[1]] = 255
		#print('coords are', out_min)
		#out_depth.write(out_filtered)
		
		#concerned with column values - col = out_min[1]
		#divide into four regions 
		# 0 - 56 # 57 - 112 # 113 - 168 # 169 - 224
		
		quad = [[],[],[],[]]
		num_frames = 1
		
		# still having issues even with only one frame - check that a min is being found on every frame with if statement above
		
		columns = out_min[1]
		if any((col >= 0 and col <= 56) for col in columns):
			quad[0].append('1')
			if len(quad[0]) == num_frames:
				print('left')
				quad[0] = []
		elif any((col > 56 and col <= 112) for col in columns):
			quad[1].append('1')
			if len(quad[1]) == num_frames:
				print('midleft')
				quad[1] = []
		elif any((col > 112 and col <= 168) for col in columns):
			quad[2].append('1')
			if len(quad[2]) == num_frames:
				print('midright')
				quad[2] = []
		elif any((col > 168 and col <= 224) for col in columns):
			quad[3].append('1')
			if len(quad[3]) == num_frames:
				print('right')
				quad[3] = []
		else:
			quad = [[],[],[],[]]
				

		#cv2.imshow('Depth Map Output', out_filtered)
		    

		end = time.time()

		print('Current FPS:', round(1/(end-start),3))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break # CTRL + Q to stop

	cap.release()
	cv2.destroyAllWindows()
	
	return


if __name__ == '__main__': 
    main()
