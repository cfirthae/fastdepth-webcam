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


vidExt = input('Video file name?\n>>') # to be added to video file name
num_frames = int(input('Number of frames?\n>>')) # num of frames an object must appear before sending move cmd
net = jetson.inference.segNet("fcn-resnet18-sun-640x512") # load segNet
rng = np.random.default_rng()
ardLeft = "http://192.168.110.218/lgd/199"
ardMid = "http://192.168.110.218/lgd/919"
ardRight = "http://192.168.110.218/lgd/991"
try:
	requests.get(url = ardLeft)
except:
	pass

try:
	requests.get(url = ardRight)
except:
	pass


# set Gstreamer pipeline - regular cv2.VideoCapture(0) doesnt work for RPi v2

def gstreamer_pipeline(
    capture_width=224,
    capture_height=224,
    display_width=224,
    display_height=224,
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
	fps_list = [] # for fps averaging 
	counter = [] # object detection counter
	lane = 'middle' # begin in middle lane


	cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
	class_mask = None
	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	print('width ', width)
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	print('height ', height)
	fps = cap.get(cv2.CAP_PROP_FPS)

	fourccDepth = cv2.VideoWriter_fourcc(*'MJPG')
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	
	# defines VideoWriter objs - final bool is essential and denotes whether video is color or not
	colorOut = cv2.VideoWriter('color_' + vidExt + '.avi',fourcc,fps,(224,224),True)
	depthOut = cv2.VideoWriter('depth_' + vidExt + '.avi',fourccDepth,fps,(224,224),False)	
	
	i = 0;
	while cap.isOpened():	# bool if vid cap is working	
		start = time.time()
		ret, frame = cap.read()
		#cv2.imshow('frame',frame)
		#colorOut.write(frame)

		image = Image.fromarray(frame) #Image.open('image.jpg') # loads PIL image from captured frame   
		image = image.resize((224,224),Image.ANTIALIAS) # resize to 224x224 with AA filtering
		    
		img_resize = np.array(image) # convert PIL to np array

		transform = transforms.Compose([transforms.ToTensor()]) 
		depth_img = transform(image) # uses above function to make resized image into pytorch tensor
		    
		    ### Segmentation Section ###
		seg_img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB) 
		seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2RGBA).astype(np.float32) # color conversions to correct segmentation input


		seg_img = jetson.utils.cudaFromNumpy(seg_img) # convert from np array to cuda

		net.Process(seg_img) # process img in model

		class_mask = jetson.utils.cudaAllocMapped(width=224, height=224, format='gray8')
		net.Mask(class_mask,224,224) # create and assign mask array of class IDs

		class_mask_np = jetson.utils.cudaToNumpy(class_mask) # cuda to np array

		class_blacklist = (1,2,13) # class ID blacklist 
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
		

		outFiltered = np.where((np.isin(class_mask,class_blacklist)),255,out)


		#find max val index using below 
		    
		outMin = np.where(outFiltered == np.amin(outFiltered))
		#print(out_min)
		
		
		#concerned with column values - col = out_min[1]
		#divide into four regions 
		# 0 - 75 # 76 - 150 # 151 - 224 

		columns = outMin[1]

		if any((col > 37 and col <= 186) for col in columns):
			counter.append('1')
			if len(counter) == num_frames:
				print('object detected in lane ', lane)
				if lane == 'middle':
					if (rng.integers(10) % 2) == 0:
						try:
							requests.get(url = ardLeft)
						except:
							pass
						lane = 'left'
						print('changed lane to ', lane)
						counter = []
						continue
					else:
						try:
							requests.get(url = ardRight)
						except:
							pass						
						lane = 'right'
						print('changed lane to ', lane)
						counter = []
						continue
				if lane == 'right':
					try:
						requests.get(url = ardLeft)
					except:
						pass
					lane = 'middle'
					print('changed lane to ', lane)
					quad = []
					continue
				if lane == 'left':
					try:
						requests.get(url = ardLeft)
					except:
						pass
					lane = 'middle'
					print('changed lane to ', lane)
					counter = []
					continue
		else:
			counter = []
		cv2.imshow('Filtered', outFiltered)		
		out[outMin[0],outMin[1]] = 255
		out[:,37] = 255
		out[:,186] = 255
		#depthOut.write(out)
		cv2.imshow('Depth Map Output', out)
		if i == 0:
			print('>> Running.')
			i = 1
		  

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
