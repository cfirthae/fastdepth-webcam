import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True



from torchvision import datasets, transforms
from PIL import Image
import cv2
import time


# set Gstreamer pipeline - regular cv2.VideoCapture(0) doesnt work for RPi v2

def gstreamer_pipeline(
    capture_width=224,
    capture_height=224,
    display_width=224,
    display_height=224,
    framerate=30,
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
	fourccDepth = cv2.VideoWriter_fourcc(*'X264')
	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	print('width ', width)
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	print('height ', height)
	fps = cap.get(cv2.CAP_PROP_FPS)

	depthOut = cv2.VideoWriter('depth.avi',fourccDepth,fps,(int(width),int(height)))
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
