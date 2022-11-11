import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
#from keras_preprocessing.image import load_img
#from keras_preprocessing.image import img_to_array

def random_crop(img_as_array, target_size=(255,255)) -> np.ndarray:
    img_h, img_w = img_as_array.shape[:2]
    target_h, target_w = target_size
    corner_h = np.random.randint(1+img_h - target_size[0])
    corner_w = np.random.randint(1+img_w - target_size[1])
    new_img = img_as_array[corner_h:corner_h+target_h,
                           corner_w:corner_w+target_w,
                           :]
    assert new_img.shape[:2] == target_size, f"{new_img.shape[:2]} != {target_size}"
    return new_img

def getStitchCropPermutation(image_as_array, perm: tuple, label_from_perm=None, tilenumberx=3, target_size=None):
    """
    Takes an image as an array and a corresponding permutations
    that agrees with tilenumberx, and returns a permuted image as an array
    """

    idx = perm
    tilesize_h = image_as_array.shape[0]//(tilenumberx)
    tilesize_w = image_as_array.shape[1]//(tilenumberx)
    channels = image_as_array.shape[2]
    if not target_size:
        target_size = (image_as_array.shape[:2])
    tiles = np.zeros(target_size[:2]+(channels,), dtype=int)
    target_tile_h = target_size[0]//tilenumberx
    target_tile_w = target_size[1]//tilenumberx
    for i, r in enumerate(idx):
        h_out_index = (i//tilenumberx)*target_tile_h
        w_out_index = (i%tilenumberx)*target_tile_w
        resize = target_size != image_as_array.shape[:2]
        reg_y = ((i+1)%tilenumberx == 0)*(target_size[0]%tilenumberx)*resize
        reg_x = (i >= (tilenumberx-1)*tilenumberx)*(target_size[1]%tilenumberx)*resize
        tiles[
            h_out_index:h_out_index+target_tile_h+reg_x,
            w_out_index:w_out_index+target_tile_w+reg_y,
            :
            ] = random_crop(image_as_array[
            (r//tilenumberx)*tilesize_h:(r//tilenumberx+1)*tilesize_h, # cutting x dim
            (r%tilenumberx)*tilesize_w:(r%tilenumberx+1)*tilesize_w,   # cutting y dim
            :                                                          # keep channels
            ], target_size=(target_tile_h+reg_x,target_tile_w+reg_y))
    out = np.array(tiles, dtype=int)

    if label_from_perm:
        label = label_from_perm[idx]
        return out, label 
    
    return out, perm

class CropTool():
    def __init__(self, target_size=None) -> None:
        self.target_size = target_size
    def __call__(self, image, perm: tuple, label_from_perm=None, tilenumberx=3):
        return getStitchCropPermutation(image, perm, label_from_perm=label_from_perm, tilenumberx=tilenumberx, target_size=self.target_size)

def main():
    from pathlib import Path
    PATH = Path("./data_test/Plantdoc/Apple_rust_leaf_test/20130519cedarapplerust.jpg")
    img1 = load_img(PATH, target_size=(255,255))
    img_data1 = img_to_array(img1, dtype = int)
    plt.imshow(img_data1)
    plt.show()
    plt.imshow(random_crop(img_data1, target_size=(224,224)))
    plt.show()
    plt.imshow(getStitchCropPermutation(img_data1, (1,2,3,4,5,6,8,0,7))[0])
    plt.show()
    plt.imshow(getStitchCropPermutation(img_data1, (1,2,3,4,5,6,8,0,7),  target_size=(224,224))[0])
    plt.show()
    PATH = Path("./data_test/Plantdoc/Apple_rust_leaf_test/20130519cedarapplerust.jpg")
    img1 = load_img(PATH, target_size=(224,224))
    img_data1 = img_to_array(img1, dtype = int)
    plt.imshow(getStitchCropPermutation(img_data1, (1,2,3,4,5,6,8,0,7))[0])
    plt.show()
    
    PATH = Path("./data_test/Plantdoc/Apple_rust_leaf_test/20130610_110514.jpg")
    img1 = tf.keras.utils.load_img(PATH, keep_aspect_ratio=True, target_size=(255,255))
    img2 = load_img(PATH)
    img_data1 = img_to_array(img1, dtype = int)
    plt.imshow(img_data1)
    plt.show()
    plt.imshow(random_crop(img_data1))
    plt.show()
    img_data2 = img_to_array(img2, dtype = int)
    plt.imshow(img_data2)
    #plt.show()
    plt.imshow(random_crop(img_data2))
    plt.show()





if __name__=='__main__':
    main()