import os
import numpy as np
from imageio import imread, imwrite
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm

def crop_border(img):
    "crop black borders of images, relevant for Montgomery images"
    mask = img > 0
    # Keeps rows and columns of images if they are not completely black
    return img[np.ix_(mask.any(1), mask.any(0))]

def find_edge(img):
    "finds the larger edge for cropping"
    y, x = img.shape
    return y if y < x else x

def crop_image(img, size):
    "Crops the image size so can extract the central part"
    y, x = img.shape
    startx = (x - size) // 2
    starty = (y - size) // 2
    
    return img[starty:starty + size, startx: startx + size]

def preprocess_full(imgdir, outdir, size=512):
    "full preprocessing, takes input and output directory"
    files = sorted(os.listdir(imgdir))
    num_imgs = len(files)

    for i, file in enumerate(tqdm(files)):

        input_path = os.path.join(imgdir, file)
        output_path = os.path.join(outdir, file)
        img = imread(input_path, as_gray=True)

        img_clean_border = crop_border(img)
        edge = find_edge(img_clean_border)
        cropped_image = crop_image(img_clean_border, edge)
        final_img = resize(cropped_image, (size,size), order=3)
       
        final_img = img_as_ubyte(final_img/255.0)
        imwrite(output_path, final_img)

if __name__ == '__main__':
    preprocess_full('Images/Images', 'Images/Cropped')
