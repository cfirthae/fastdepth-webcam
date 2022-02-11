import numpy as np

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
        "video/x-raw(meqmory:NVMM), "
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
	pth = torch.load('best_result.pth.tar') # load fastdepth model w Torch
	model = pth['model'] # index correct model path
	quad = [[],[],[],[]] # object detection counter
	fps_list = [] # for fps averaging 
	#model.eval()


	cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER) # initialize webcam
	class_mask = None
	#cap = cv2.VideoCapture('chair_approach_color.avi') # load video 
	
	# Define the codec and create VideoWriter object
	#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	#out_depth = cv2.VideoWriter('depth_out.avi',fourcc,10,(224,224),False)

	while cap.isOpened():	# bool if vid cap is working	
		start = time.time()
		ret, frame = cap.read()
		cv2.imshow('frame',frame)


		image = Image.fromarray(frame) #Image.open('image.jpg') # loads PIL image from captured frame
		#print('size is',image.size)    
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

		#net.Overlay(seg_img)
		#jetson.utils.cudaDeviceSynchronize()

		#seg_img = jetson.utils.cudaToNumpy(seg_img)

		#seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGBA2BGR).astype(np.float32)

		#cv2.imwrite("seg.jpeg",seg_img)  
		#cv2.imshow('Segmentation Output', seg_img/255)           

		    ### Depth Map Section ###
		x = depth_img.resize(1,3,224,224)
		    #x = torch.rand(1,3,224,224)
		x_torch = x.type(torch.cuda.FloatTensor)


		depth = model(x_torch) #returns torch.Tensor of shape torch.Size([1,1,224,224])
		#the above line takes the longest to run and is the result of the first frame wait time
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
