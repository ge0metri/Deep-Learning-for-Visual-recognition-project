import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from itertools import permutations

from keras.preprocessing.image import Iterator
import random


def showPermImg(X, perm, tilenum = 3):
    plt.figure(figsize=(tilenum,tilenum))
    for i in range(tilenum**2):
        plt.subplot(tilenum,tilenum,i+1)
        plt.imshow(X[i])
        plt.xticks([]), plt.yticks([])
        plt.title(int(perm[i]))

def PermMapToOneHot(num, number_of_different_perms):
    """
    creates a subset of size "number_of_different_perms" of permutations the range of "num"
    """
    assert int(num) < 10, "Please choose a smaller number of tiles!"
    assert number_of_different_perms <= np.math.factorial(num), "There are not that many permutations!"
    all_permutations = list(permutations(range(num)))
    perms = random.sample(all_permutations, number_of_different_perms) # without replacement 
    eye = np.eye(number_of_different_perms)
    return {perms[i]:eye[i] for i in range(number_of_different_perms)}
    
def getSubsetPermutation(image_as_array, perm: tuple, label_from_perm=None, tilenumberx=3):
    """
    Takes an image as an array and a corresponding permutations
    that agrees with tilenumberx, and returns a permuted subset image as an array
    args: 
        perm: tuple of tilnumberx numbers permutated, plus a number if horizontal or not
              i.e. [1,0,2,1] would mean horizantal three tiles would be mixed. 
                   [0,0,2,1] would mean vertical three tiles would be mixed.
    """

    idx = perm[1:]
    assert len(idx) == tilenumberx, f"{len(idx)} is not equal {tilenumberx}"
    vert_not_hori = perm[0]
    tilesize_h = image_as_array.shape[0]//(tilenumberx)
    tilesize_w = image_as_array.shape[1]//(tilenumberx)
    
    if vert_not_hori:
        tiles = [
            image_as_array[
                (i)*tilesize_h:(i+1)*tilesize_h, # cutting x dim
                (image_as_array.shape[1]//3):(image_as_array.shape[1]//3) + tilesize_w,   # cutting middle y dim
                :                                                          # keep channels
                ]
                for i in idx
            ]
    else:
        tiles = [
            image_as_array[
                (image_as_array.shape[0]//3):(image_as_array.shape[0]//3)+tilesize_h, # cutting middle x dim
                (i)*tilesize_w:(i+1)*tilesize_w,   # cutting y dim
                :                                                          # keep channels
                ]
                for i in idx
            ]



    out = np.array(tiles)

    if label_from_perm:
        label = label_from_perm[idx]
        return out, label 
    
    return out, perm

def getPermutation(image_as_array, perm: tuple, label_from_perm=None, tilenumberx=3):
    """
    Takes an image as an array and a corresponding permutations
    that agrees with tilenumberx, and returns a permuted image as an array
    """

    idx = perm
    tilesize_h = image_as_array.shape[0]//(tilenumberx)
    tilesize_w = image_as_array.shape[1]//(tilenumberx)

    tiles = [
        image_as_array[
            (i//tilenumberx)*tilesize_h:(i//tilenumberx+1)*tilesize_h, # cutting x dim
            (i%tilenumberx)*tilesize_w:(i%tilenumberx+1)*tilesize_w,   # cutting y dim
            :                                                          # keep channels
            ]
            for i in idx
        ]

    out = np.array(tiles)

    if label_from_perm:
        label = label_from_perm[idx]
        return out, label 
    
    return out, perm

class PermOneHotDataGen(Iterator):
    def __init__(self, input, batch_size=64,
                 preprocess_func=None, 
                 reuse = 1, tilenumberx = 3,
                 shuffle_permutations=False, shuffle=True):

        
        self.tilenumberx = tilenumberx
        self.batch_size = batch_size
        self.preprocess_func = preprocess_func
        self.shuffle_permutations = shuffle_permutations
        self.shuffle = shuffle 

        self.size_of_image = (255,255)

        self.number_of_tiles = tilenumberx**2

        if isinstance(input, list):
            self.im_as_files = True
            self.input_shape = (
                self.number_of_tiles, 
                self.size_of_image[0]//tilenumberx, # x dimension
                self.size_of_image[1]//tilenumberx, # y dimension
                3                                   # number of channels
                )

            self.images = input * reuse
        else:
            self.input_shape = self.images.shape[1:]
            self.images = input
        
        self.number_of_images = len(self.images)
        self.number_of_different_perms = self.number_of_tiles
        self.PermDict = None
        self.perms = [None]*self.number_of_images
        if not self.shuffle_permutations:
            self.max_perms = 26
            self.number_of_different_perms = min(self.max_perms, np.math.factorial(self.number_of_tiles))
            self.PermDict = PermMapToOneHot(self.number_of_tiles, self.number_of_different_perms)
            self.ReverseDict = {tuple(val):key for (key, val) in self.PermDict.items()}
            self.perms_labels = random.choices(list(self.PermDict.items()), k=self.number_of_images) # with replacement
            self.perms = [perm_label[0] for perm_label in self.perms_labels] # actual permutation
            self.labels = [perm_label[1] for perm_label in self.perms_labels] # corresponding label 

        # add dimension if the images are greyscale
        if len(self.input_shape) == 2:
            self.input_shape = self.input_shape + (1,)


        super(PermOneHotDataGen, self).__init__(
            self.number_of_images, batch_size, shuffle_permutations, None)


    def get_perm_from_label(self, label):
        assert self.PermDict, "There is no PermDict"
        if not isinstance(label, tuple):
            label = tuple(label)
        return self.ReverseDict[label]
        

    def test_data_gen(self):
        print(self.PermDict[self.perms[0]])
        print(self.perms[0])
        print(self.labels[0]) 


    def _get_batches_of_transformed_samples(self, index_array):

        # create array to hold the images
        batch_x = np.zeros((len(index_array), ) + self.input_shape, dtype='float32')

        # create array to hold the labels
        batch_y = np.zeros((len(index_array), self.number_of_different_perms), dtype='float32')



        # iterate through the current batch
        for i, j in enumerate(index_array):
            
            if self.im_as_files:
                image = img_to_array(
                    load_img(self.images[j], target_size=self.size_of_image)
                    ) / 255 # should prob not be hardcoded
            else:
                image = self.images[j].squeeze()

            if self.preprocess_func:
                image = self.preprocess_func(image)


            if self.shuffle_permutations:
                perm = np.random.permutation(range(self.number_of_tiles))
            else:
                perm = self.perms[j]

            X, y = getPermutation(image, perm, self.PermDict, tilenumberx=self.tilenumberx)
            # permute image according to perm
            # store the image and label in their corresponding batches
            batch_x[i] = X
            batch_y[i] = y

        return batch_x, batch_y

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)



class PermSubsetDataGen(PermOneHotDataGen):
    def __init__(self, input, batch_size=64,
                 preprocess_func=None, 
                 reuse = 1, tilenumberx = 3,
                 shuffle_permutations=False, shuffle=True, vert=False):
        self.vert = vert
        super(PermSubsetDataGen, self).__init__(input, batch_size,
                 preprocess_func, 
                 reuse, tilenumberx,
                 shuffle_permutations, shuffle)
        self.number_of_different_perms = self.tilenumberx
        self.input_shape = (
                self.tilenumberx, 
                self.size_of_image[0]//tilenumberx, # x dimension
                self.size_of_image[1]//tilenumberx, # y dimension
                3                                   # number of channels
                )

    def _get_batches_of_transformed_samples(self, index_array):

        # create array to hold the images
        batch_x = np.zeros((len(index_array), ) + self.input_shape, dtype='float32')

        # create array to hold the labels
        size = self.number_of_different_perms + 1
        if self.vert:
            size = self.number_of_different_perms
        batch_y = np.zeros((len(index_array), size), dtype='float32')



        # iterate through the current batch
        for i, j in enumerate(index_array):
            
            if self.im_as_files:
                image = img_to_array(
                    load_img(self.images[j], target_size=self.size_of_image)
                    ) / 255 # should prob not be hardcoded
            else:
                image = self.images[j].squeeze()

            if self.preprocess_func:
                image = self.preprocess_func(image)

            perm = np.zeros(self.tilenumberx + 1, dtype=int)
            perm[0] = np.random.randint(2)
            if self.vert:
                perm[0] = 1
            perm[1:] = np.random.permutation(self.tilenumberx)


            X, y = getSubsetPermutation(image, perm, tilenumberx=self.tilenumberx)
            # permute image according to perm
            # store the image and label in their corresponding batches
            batch_x[i] = X
            batch_y[i] = y[1:]

        return batch_x, batch_y




def main():
    import os 
    from pathlib import Path
    PATH = Path('data_test/plantvillage/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG') 
    tilenumberx = [2,3]
    for i in tilenumberx:
        datagenerator = PermOneHotDataGen(input=[PATH],batch_size=1,tilenumberx=i)
        datagenerator.test_data_gen()
        X,y = datagenerator.next()
        print(y)
        created_label = datagenerator.get_perm_from_label(tuple(y[0]))
        print(created_label)
        showPermImg(X[0], created_label, i)
        plt.show()

    #new test of subset
    for i in tilenumberx:
        print(f'======== Subset test with {i} tiles =========')
        datagenerator = PermSubsetDataGen(input=[PATH],batch_size=1,tilenumberx=i, vert=True)
        X,y = datagenerator.next()
        print(y)
        plt.figure(figsize=(i,1))
        for k in range(i):
            plt.subplot(i,i,k+1)
            plt.imshow(X[0][k])
            plt.xticks([]), plt.yticks([])
        plt.show()

    img1 = load_img(PATH, target_size=(255,255))
    img_data1 = img_to_array(img1, dtype = int)
    PermutationDict = PermMapToOneHot(4,24)    
    ReverseDict = {tuple(val):key for (key, val) in PermutationDict.items()}
    number_tiles_x = 2
    perm = (2,3,1,0)
    assert len(perm) == number_tiles_x**2, "Perm does not match number of tiles!"
    permuted_image, label = getPermutation(img_data1, perm, PermutationDict, tilenumberx=number_tiles_x)
    showPermImg(permuted_image, ReverseDict[tuple(label)], tilenum=number_tiles_x)
    plt.show()




if __name__=='__main__':
    main()