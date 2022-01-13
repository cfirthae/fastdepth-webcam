import cv2
import jetson.inference
import jetson.utils
from segnet_utils import *
import numpy as np
from PIL import Image
import time


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

net = jetson.inference.segNet("fcn-resnet18-sun-512x400")
net.SetOverlayAlpha(150.0)
camera = cv2.VideoCapture("full_hallway_color.avi")
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
img_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
img_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = camera.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('full_hallway_segmented.avi',fourcc,10,(int(img_width),int(img_height)))

class_mask = None
#class_mask_np = None

while camera.isOpened():
        _,image = camera.read()
        

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA).astype(np.float32)

        
        img_input = jetson.utils.cudaFromNumpy(img)

        net.Process(img_input)
        #jetson.utils.cudaDeviceSynchronize()
        grid_width, grid_height = net.GetGridSize()
        if class_mask is None:
                class_mask = jetson.utils.cudaAllocMapped(width=224, height=224, format=img_input.format)
                net.Mask(class_mask,224,224) #change mask size here and above

                class_mask_np = jetson.utils.cudaToNumpy(class_mask)[:,0:2]
        print('Mask shape is',class_mask_np.shape)
        #print(class_mask_np)

 
        net.Mask(img_input)
        jetson.utils.cudaDeviceSynchronize()

        img_np = jetson.utils.cudaToNumpy(img_input)

        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR).astype(np.float32)
        #out.write(np.uint8(img_np))
        img_cv2 = img_np.copy()
        cv2.imshow('OpenCV Output', img_cv2/255)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break # CTRL + Q to stop


camera.release()
cv2.destroyAllWindows()


### extra
#x = depth_img.resize(1,3,224,224)
#            #x = torch.rand(1,3,224,224)
        #    x_torch = x.type(torch.cuda.FloatTensor)
           
       #     depth = model(x_torch) #returns torch.Tensor of shape torch.Size([1,1,224,224])

      #      depth_min = depth.min()
     #       depth_max = depth.max()
    #        max_val = (2**(8))-1 # 255

   #         if depth_max - depth_min > np.finfo("float").eps:
  #              out = max_val * (depth - depth_min) / (depth_max - depth_min)
                #returns torch.Tensor of shape torch.Size([1,1,224,224])
 #           else:
#                out = np.zeros(depth.shape, dtype=depth.type)

#            out = out.cpu().detach().numpy()  
#            out = out.reshape(224,224)  
            
#            out = Image.fromarray(out) # creates PIL Image obj from above array
#            out = out.convert('L')  # converts image to grayscale 
            
#            out = np.array(out)
            #out_depth.write(out)

#            cv2.imshow('Depth Map Output', out)
