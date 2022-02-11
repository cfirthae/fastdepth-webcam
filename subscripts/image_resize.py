import cv2
from PIL import Image

frame = cv2.imread('stock_hallway.jpg')
img_input = Image.fromarray(frame)

image = img_input.resize((224,224),Image.ANTIALIAS)

image.save('resize.jpg')

