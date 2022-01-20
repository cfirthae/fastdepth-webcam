import numpy as np
import time

start = time.time()

def mask_image(image, mask, masked_value):
    
    scale_factor = image.shape[0]//mask.shape[0] # how much the mask needs to be scaled up by to match the image's size
    
    resized_mask = mask.repeat(scale_factor,axis=0).repeat(scale_factor,axis=1)

    return(np.where((resized_mask==0),masked_value,image)) # where the mask==0, return the masked value, else return the value of the original array.

def seg_filter( a, blacklist ):
    s = a.shape
    l = np.ndarray.flatten( a )
    ol = [ e not in blacklist for e in l ]
    return np.reshape( ol, s ).astype(int)



mask = np.random.randint(0,10,size=[16,16])
blacklist = (0,1,2,13)
image = np.random.randint(0,254, size=[224,224])
bool_mask = seg_filter(mask, blacklist) # 1's are to be analyzed, 0's are not

final = mask_image(image, bool_mask, 255)

end = time.time()

print('fps is',  1/(end-start))
