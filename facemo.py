import numpy as np
from PIL import Image, ImageOps

def load_image(source : str) -> np.ndarray:
    ''' Opens an image from specified source and returns a numpy array with image rgb data
    '''
    with Image.open(source) as im:
        im_arr = np.asarray(im)
    return im_arr

def resize_image(img : Image, size : tuple) -> np.ndarray:
    '''Takes an image and resizes to a given size (width, height) as passed to the size parameter
    '''
    resz_img = ImageOps.fit(img, size, Image.ANTIALIAS, centering=(0.5, 0.5))
    return np.array(resz_img)