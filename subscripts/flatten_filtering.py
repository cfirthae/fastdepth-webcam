import numpy as np
import time 
def seg_filter( a, blacklist ):
    s = a.shape
    l = np.ndarray.flatten( a )
    ol = [ e not in blacklist for e in l ]
    return np.reshape( ol, s ).astype(int)

def img_filter(image,bool_mask):
    s = image.shape
    image_f = np.ndarray.flatten(image)
    mask_f = np.ndarray.flatten(bool_mask)
    concat = np.stack((image_f,mask_f)) # image stacked on mask
    for index, item in enumerate(concat[0]):
        if concat[1][index] == 0:
            concat[0][index] = 255
        else:
            pass
    return np.reshape(concat[0],s)

start = time.time()

mask = np.random.randint(1,10,size=[224,224])
blacklist = (1,3,5)
image = np.random.randint(0,254,size=[224,224])
      
bool_mask = seg_filter(mask, blacklist) # 1's are to be analyzed, 0's are not
filtered_image = img_filter(image,bool_mask)

end = time.time()
