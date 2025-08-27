"""
Tries to remove unnecessary black borders/background around the images,
and "trim" the images so that they take up the entirety of the image.
"""

import os
import numpy as np
from PIL import Image
import warnings
from multiprocessing import Pool
from tqdm import tqdm
import cv2

def trim(im):
    """
    Converts the image to grayscale using cv2, then computes binary matrix
    of the pixels that are above a certrain threshold, then takes out the first
    row where a certain percentage of the pixels are above the threshold will
    be the first clip point. Same idea for col, max row, max col.
    """

    percentage = 0.02

    img = np.array(im)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = img_gray > 0.1 * np.mean(img_gray[img_gray != 0]) # Checks which pixels are brighter than 10% of the mean of the non-black pixels
    row_sums = np.sum(im, axis=1)
    col_sums = np.sum(im, axis=0)
    rows = np.where(row_sums > img.shape[1] * percentage)[0]
    cols = np.where(col_sums > img.shape[0] * percentage)[0]
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)
    im_crop = img[min_row : max_row+1, min_col : max_col+1]
    return Image.fromarray(im_crop)

def resize_maintain_aspect(image, desired_size):
    """
    Resizes an image while maintaining the aspect ratio.
    """

    old_size = image.size # old_size[0] is in (width, height) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple((int(x * ratio) for x in old_size))
    im = image.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im

def save_single(args):
    img_file, input_path_folder, output_path_folder, output_size = args
    image_original = Image.open(os.path.join(input_path_folder, img_file))
    image = trim(image_original)
    image = resize_maintain_aspect(image, desired_size=output_size[0])
    image = image.resize(output_size) # Might not be necessary
    image.save(os.path.join(output_path_folder + img_file))

def fast_image_resize(input_path_folder, output_path_folder, output_size=None):
    """
    Uses multiprocessing to make it fast.
    """

    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=100")
        exit()

    if not os.path.exists(output_path_folder):
        os.makedirs(output_path_folder)
    
    jobs = [
        (file, input_path_folder, output_path_folder, output_size)
        for file in os.listdir(input_path_folder)
    ]

    with Pool() as p:
        list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))

if __name__ == "__main__":
    fast_image_resize('/home/haochen/Seagate8T/universe/DataSet/diabet_fundus/aptos2019-blindness-detection/test_images/',
    '/home/haochen/Seagate8T/universe/DataSet/diabet_fundus/aptos2019-blindness-detection/test_images_preprocessed1024/',
    output_size=(1024, 1024))
    
    fast_image_resize('/home/haochen/Seagate8T/universe/DataSet/diabet_fundus/aptos2019-blindness-detection/train_images/',
    '/home/haochen/Seagate8T/universe/DataSet/diabet_fundus/aptos2019-blindness-detection/train_images_preprocessed1024/',
    output_size=(1024, 1024))
    
    fast_image_resize('/home/haochen/Seagate8T/universe/DataSet/diabet_fundus/EyePacs/train/',
    '/home/haochen/Seagate8T/universe/DataSet/diabet_fundus/EyePacs/train_preprocessed1024/',
    output_size=(1024, 1024))

    fast_image_resize('/home/haochen/Seagate8T/universe/DataSet/diabet_fundus/EyePacs/test/',
    '/home/haochen/Seagate8T/universe/DataSet/diabet_fundus/EyePacs/test_preprocessed1024/',
    output_size=(1024, 1024))