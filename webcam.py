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
args = utils.parse_command()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # Set the GPU.

fieldnames = ['rmse', 'mae', 'delta1', 'absrel',
            'lg10', 'mse', 'delta2', 'delta3', 'data_time', 'gpu_time']
best_fieldnames = ['best_epoch'] + fieldnames
best_result = Result()
best_result.set_to_worst()

net = jetson.inference.segNet("fcn-resnet18-sun-512x400") # load segNet

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
    global args, best_result, output_directory, train_csv, test_csv

    # Data loading code
    print("=> creating data loaders...")

    # evaluation mode
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no model found at '{}'".format(args.evaluate)
        print("=> loading model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        # print(checkpoint)
        if type(checkpoint) is dict:
            args.start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            model = checkpoint['model']
            print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        else:
            model = checkpoint
            args.start_epoch = 0
        #print(model)

        model.eval()

        
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        class_mask = None
        #cap = cv2.VideoCapture('test_video.mp4')
        # Define the codec and create VideoWriter object
        #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        #height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #fps = cap.get(cv2.CAP_PROP_FPS)

        #out_depth = cv2.VideoWriter('depth_out.avi',fourcc,10,(224,224),False)
        #out_color = cv2.VideoWriter('color_out.avi',fourcc,10,(int(width),int(height)))

        print('=> Capturing video using pipeline', gstreamer_pipeline(flip_method=0))
        while True:


            start = time.time()
            ret, frame = cap.read()


            image = Image.fromarray(frame) # loads PIL image from captured frame
            
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
            if class_mask is None:
                    class_mask = jetson.utils.cudaAllocMapped(width=224, height=224, format='gray8')
                    net.Mask(class_mask,224,224)

                    class_mask_np = jetson.utils.cudaToNumpy(class_mask)
            print('Mask shape is',class_mask_np.shape)


 
            #net.Overlay(seg_img)
            jetson.utils.cudaDeviceSynchronize()

            #seg_img = jetson.utils.cudaToNumpy(seg_img)

            #seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGBA2BGR).astype(np.float32)

            
            #cv2.imshow('Segmentation Output', seg_img/255)           

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

            #find max val index using below 
            
            #out_min = np.where(out == np.amin(out))
            #out[out_min[0],out_min[1]] = 255
            #out_depth.write(out)

            #cv2.imshow('Depth Map Output', out)
            

            end = time.time()
            print('out shape is',out.shape)
            print('Current FPS:', round(1/(end-start),3))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break # CTRL + Q to stop


            #out.save('depth.png')
        
        cap.release()
        cv2.destroyAllWindows()

        #output_directory = os.path.dirname(args.evaluate)
        #validate(val_loader, model, args.start_epoch, write_to_file=False)
        return


if __name__ == '__main__': 
    main()
