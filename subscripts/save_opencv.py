import numpy as np
import cv2

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

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print('width ', width)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('height ', height)
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('output.avi',fourcc,fps,(int(width),int(height)))

while cap.isOpened():
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)
        #print('frame shape is', frame.shape)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
