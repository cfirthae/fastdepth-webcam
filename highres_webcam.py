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

def gstreamer_pipeline(
    capture_width=448,
    capture_height=448,
    display_width=448,
    display_height=448,
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
	model.eval()
	imsize = 448
	quad = [[],[],[],[]]
	fps_list = []


	cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
	class_mask = None
	#cap = cv2.VideoCapture('full_hallway_color.avi')
	# Define the codec and create VideoWriter object
	#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	#width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	#height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	#fps = cap.get(cv2.CAP_PROP_FPS)

	#out_depth = cv2.VideoWriter('depth_out.avi',fourcc,10,(224,224),False)
	#out_color = cv2.VideoWriter('color_out.avi',fourcc,10,(int(width),int(height)))
	
	while cap.isOpened():
		start = time.time()
		ret, frame = cap.read()
		#cv2.imshow('frame', frame)


		image = Image.fromarray(frame) #Image.open('image.jpg') # loads PIL image from captured frame
		    
		image = image.resize((imsize,imsize),Image.ANTIALIAS) # resize to 224x224 with AA filtering
		   
		transform = transforms.Compose([transforms.ToTensor()]) 
		depth_img = transform(image) # uses above function to make resized image into pytorch tensor
		img_resize = np.array(image)
		    ### Segmentation Section ###
		seg_img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
		seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2RGBA).astype(np.float32)


		seg_img = jetson.utils.cudaFromNumpy(seg_img)

		net.Process(seg_img)
		    
		#grid_width, grid_height = net.GetGridSize()

		class_mask = jetson.utils.cudaAllocMapped(width=imsize, height=imsize, format='gray8')
		net.Mask(class_mask,imsize,imsize)

		class_mask_np = jetson.utils.cudaToNumpy(class_mask)

		class_blacklist = (1,2,13)
		class_mask = np.reshape(class_mask_np, [imsize,imsize])

		net.Overlay(seg_img)
		jetson.utils.cudaDeviceSynchronize()

		seg_img = jetson.utils.cudaToNumpy(seg_img)

		seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGBA2BGR).astype(np.float32)

		#cv2.imwrite("seg.jpeg",seg_img)  
		#cv2.imshow('Segmentation Output', seg_img/255)           

		    ### Depth Map Section ###
		x = depth_img.resize(1,3,imsize,imsize)
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
		out = out.reshape(imsize,imsize)  
		    
		out = Image.fromarray(out) # creates PIL Image obj from above array
		out = out.convert('L')  # converts image to grayscale 
		    
		out = np.array(out)

		#out_filtered = np.where((np.isin(class_mask,class_blacklist)),255,out)
		#print('filtered out shape is', out_filtered.shape)


		    #find max val index using below 
		    
		#out_min = np.where(out_filtered == np.amin(out_filtered))
		#out_filtered[out_min[0],out_min[1]] = 255
		#cv2.imwrite("depth448.jpeg",out)

		#cv2.imshow('Depth Map Output', out)
		out_filtered = np.where((np.isin(class_mask,class_blacklist)),255,out)


		    #find max val index using below 
		    
		out_min = np.where(out_filtered == np.amin(out_filtered))
		#out_depth.write(out_filtered)
		
		#concerned with column values - col = out_min[1]
		#divide into four regions 
		# 0 - 56 # 57 - 112 # 113 - 168 # 169 - 224
		check = []
		num_frames = 3
		
		
		
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
		#print('quad pre check is',quad)
		for ele in quad:
			if len(ele) > 0:
				check.append(ele)
				if len(check) > 1:
					quad = [[],[],[],[]]				
		#print('quad post check is',quad)
		cv2.imshow('Depth Map Output', out_filtered)

		end = time.time()

		fps_list.append(round(1/(end-start),3))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			print('avg fps was', sum(fps_list)/len(fps_list))
			break # CTRL + Q to stop

	cap.release()
	cv2.destroyAllWindows()

	return


if __name__ == '__main__': 
    main()
