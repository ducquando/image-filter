import cv2
import numpy as np
import streamlit as st
import glob, os
import math
import random
from scipy import spatial
from PIL import Image
from facemo import load_image, resize_image

@st.cache_data
def bw_filter(img):
    img_gray = img.copy()

    # Convert into grayscale color mode
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    return img_gray

@st.cache_data
def vignette(img, level = 2):
    height, width = img.shape[:2]  

    # Generate vignette mask using Gaussian kernels.
    X_resultant_kernel = cv2.getGaussianKernel(width, width / level)
    Y_resultant_kernel = cv2.getGaussianKernel(height, height / level)
        
    # Generating resultant_kernel matrix.
    # H x 1 * 1 x W
    kernel = Y_resultant_kernel * X_resultant_kernel.T 
    mask = kernel / kernel.max()
    
    img_vignette = img.copy()
        
    # Applying the mask to each channel in the input image.
    for i in range(3):
        img_vignette[:,:,i] = img_vignette[:,:,i] * mask

    return img_vignette

@st.cache_data
def sepia(img):
    img_sepia = img.copy()

    # Converting to RGB as sepia matrix below is for RGB.
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB) 
    img_sepia = np.array(img_sepia, dtype = np.float64)
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.393, 0.769, 0.189],
                                                    [0.349, 0.686, 0.168],
                                                    [0.272, 0.534, 0.131]]))
    
    # Clip values to the range [0, 255].
    img_sepia = np.clip(img_sepia, 0, 255)
    img_sepia = np.array(img_sepia, dtype = np.uint8)
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)

    return img_sepia

@st.cache_data
def pencil_sketch(img, ksize = 5):
    img_sketch = img.copy()

    # Generate a Gaussian-blur-filtered image with auto-calculated sigma value
    img_blur = cv2.GaussianBlur(img_sketch, (ksize, ksize), 0, 0)
    img_sketch, _ = cv2.pencilSketch(img_blur)

    return img_sketch

@st.cache_data
def gaussian_noise(img, var = 0.2):
    img_noise = img.copy()
    
    gaussian = np.random.normal(loc = 0, scale = 1, size = img_noise.shape)

    # Add random number to the image to create noisy effect
    img_noise = np.clip((img_noise * (1 + gaussian * var)), 0, 255)
        
    img_noise = img_noise.astype(np.uint8)

    return img_noise

@st.cache_data
def mosaic(img, size = 50):
    img_mosaic = img.copy()
    img_mosaic = img_mosaic[:,:,::-1]
    
    # Pixelate the image by keeping every nth pixel
    mos_template = img_mosaic[::size, ::size]
    target_res = np.array([int(img_mosaic.shape[1]/size), int(img_mosaic.shape[0]/size)])
    mosaic_size = (size, size)

    # Load reference images
    images = []
    for file in glob.glob('Images/References/*'):
        im = load_image(file)
        images.append(im)
    images = [i for i in images if i.ndim == 3]

    # Resize images and store them into a 3D numpy array
    images = [resize_image(Image.fromarray(i), (mosaic_size[0], mosaic_size[1])) for i in images]
    images_array = np.asarray(images)

    # Get RGB color, set KDT tree, and store the value of best match
    image_values = np.apply_over_axes(np.mean, images_array, [1,2]).reshape(len(images), 3)
    tree = spatial.KDTree(image_values)
    image_idx = np.zeros(target_res, dtype = np.uint32)
    for i in range(target_res[0]):
        for j in range(target_res[1]):
            template = mos_template[j, i]
            match = tree.query(template, k = 40)
            pick = random.randint(0, 39)
            image_idx[i, j] = match[1][pick]
    
    # Replace pixels with images
    canvas = Image.new('RGB', (mosaic_size[0] * target_res[0], mosaic_size[1] * target_res[1]))
    for i in range(target_res[0]):
        for j in range(target_res[1]):
            arr = images[image_idx[i, j]]
            x, y = i * mosaic_size[0], j * mosaic_size[1]
            im = Image.fromarray(arr)
            canvas.paste(im, (x,y))

    return np.asarray(canvas)