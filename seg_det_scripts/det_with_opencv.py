import cv2
import jetson.inference
import jetson.utils
import numpy as np

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

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.25)
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file
camera = cv2.VideoCapture("full_hallway_color.avi")
#camera = jetson.utils.videoSource('/dev/video0')      # '/dev/video0' for V4L2
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = camera.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('full_hallway_detection.avi',fourcc,10,(int(width),int(height)))

while camera.isOpened():
        _,image = camera.read()
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA).astype(np.float32)
        img = jetson.utils.cudaFromNumpy(img)
        detections = net.Detect(img, image.shape[1], image.shape[0])
        img_cv2 = jetson.utils.cudaToNumpy(img)
        #display.Render(img)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGBA2BGR).astype(np.float32)
        out.write(np.uint8(img_cv2))
        cv2.imshow('OpenCV Output', img_cv2/255)
        #display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break # CTRL + Q to stop
camera.release()
cv2.destroyAllWindows()
