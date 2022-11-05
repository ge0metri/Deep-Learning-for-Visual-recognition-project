import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from helper import *

from keras.preprocessing.image import Iterator



class PermOneHotDataGen(Iterator):
    def __init__(self, input, batch_size=64,
                 preprocess_func=None, 
                 reuse = 1, tilenumberx = 3,
                 shuffle_permutations=False, 
                 shuffle=True, 
                 max_perms=25, 
                 target_size=(255,255,3), 
                 stitched=False,
                 one_hot_encoding=False,
                 Permutations_dictionary=None):
                 
        self.one_hot_encoding = one_hot_encoding
        self.stitched = stitched
        self.tilenumberx = tilenumberx
        self.batch_size = batch_size
        self.preprocess_func = preprocess_func
        self.shuffle_permutations = shuffle_permutations
        self.shuffle = shuffle 

        self.size_of_image = target_size

        self.number_of_tiles = tilenumberx**2


        if isinstance(input, list):
            self.im_as_files = True
            self.input_shape = (
                self.number_of_tiles, 
                self.size_of_image[0]//tilenumberx, # height dimension
                self.size_of_image[1]//tilenumberx, # width dimension
                self.size_of_image[2]               # number of channels
                )

            self.images = input * reuse
        else:
            self.input_shape = self.images.shape[1:]
            self.images = input
        
        self.number_of_images = len(self.images)
        self.PermDict = Permutations_dictionary 
        self.perms = [None]*self.number_of_images

        self.max_perms = max_perms
        self.number_of_different_perms = min(self.max_perms, np.math.factorial(self.number_of_tiles))
        if not Permutations_dictionary:
            self.PermDict = PermMapToOneHot(self.number_of_tiles, self.number_of_different_perms)
        self.ReverseDict = {tuple(val):key for (key, val) in self.PermDict.items()}
        self.perms_labels = random.choices(list(self.PermDict.items()), k=self.number_of_images) # with replacement
        self.perms = [perm_label[0] for perm_label in self.perms_labels] # actual permutation
        self.labels = [perm_label[1] for perm_label in self.perms_labels] # corresponding label 
        if self.one_hot_encoding:
            self.label_dimension = self.number_of_different_perms
        else:
            self.label_dimension = self.number_of_tiles
        # add dimension if the images are greyscale
        if len(self.input_shape) == 2:
            self.input_shape = self.input_shape + (1,)


        super(PermOneHotDataGen, self).__init__(
            self.number_of_images, batch_size, shuffle, None)

    def get_perm_dict(self):
        return self.PermDict


    def get_perm_from_label(self, label):
        assert self.PermDict, "There is no PermDict"
        if not isinstance(label, tuple):
            label = tuple(label)
        return self.ReverseDict[label]
        

    def test_data_gen(self):
        pass

    def _get_batches_of_transformed_samples(self, index_array):

        # create array to hold the images
        if self.stitched:
            batch_x = np.zeros((len(index_array), ) + self.size_of_image, dtype='float32')
        else:
            batch_x = np.zeros((len(index_array), ) + self.input_shape, dtype='float32')
        #batch_x = np.zeros((self.number_of_tiles,) +(len(index_array), ) + self.input_shape[1:], dtype='float32')

        # create array to hold the labels
        batch_y = np.zeros((len(index_array), self.label_dimension), dtype='float32')



        # iterate through the current batch
        for i, j in enumerate(index_array):
            
            if self.im_as_files:
                image = img_to_array(
                    load_img(self.images[j], target_size=self.size_of_image)
                    ) 
            else:
                image = self.images[j].squeeze()

            if self.preprocess_func:
                image = self.preprocess_func(image)
            else:
                image = image/255


            if self.shuffle_permutations and not self.one_hot_encoding:
                perm = np.random.permutation(range(self.number_of_tiles))
            else:
                if self.one_hot_encoding:
                    perm = self.perms[j]
                else:
                    perm = self.perms[np.random.randint(len(self.perms))]
            if self.stitched:
               X, y = getStitchedPermutation(image, perm, None, tilenumberx=self.tilenumberx)
            else:
               X, y = getPermutation(image, perm, None, tilenumberx=self.tilenumberx)
            # permute image according to perm
            # store the image and label in their corresponding batches
            batch_x[i] = X
            #batch_x[:,i,:,:,:] = X
            if self.one_hot_encoding:
                y = self.PermDict[tuple(perm)]       
            batch_y[i] = y

        return batch_x, batch_y

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)



def main():

    PATH = Path('./data_test/plantvillage/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG') 
    tilenumberx = [2,3]
    for i in tilenumberx:
        datagenerator = PermOneHotDataGen(input=[PATH],batch_size=1,tilenumberx=i)
        datagenerator.test_data_gen()
        X,y = datagenerator.next()
        print(y)
        showPermImg(X[0], y[0], i)
        plt.show()

    for i in tilenumberx:
        datagenerator = PermOneHotDataGen(input=[PATH],batch_size=1,tilenumberx=i, shuffle_permutations=True)
        datagenerator.test_data_gen()
        X,y = datagenerator.next()
        print(y)
        showPermImg(X[0], y[0], i)
        plt.show()

    for i in tilenumberx:
        datagenerator = PermOneHotDataGen(input=[PATH],batch_size=1,tilenumberx=i, one_hot_encoding=True)
        datagenerator.test_data_gen()
        X,y = datagenerator.next()
        print(y)
        created_label = datagenerator.get_perm_from_label(tuple(y[0]))
        print(created_label)
        showPermImg(X[0], created_label, i)
        plt.show()

    for i in tilenumberx:
        datagenerator = PermOneHotDataGen(input=[PATH],batch_size=1,tilenumberx=i, shuffle_permutations=True, one_hot_encoding=True)
        datagenerator.test_data_gen()
        X,y = datagenerator.next()
        print(y)
        created_label = datagenerator.get_perm_from_label(tuple(y[0]))
        print(created_label)
        showPermImg(X[0], created_label, i)
        plt.show()

    for i in tilenumberx:
        datagenerator_get_dict = PermOneHotDataGen(
            input=[PATH],
            batch_size=1,
            tilenumberx=i, 
            shuffle_permutations=True, 
            one_hot_encoding=True)
        perm_dict_1 = datagenerator_get_dict.get_perm_dict() 
        datagenerator = PermOneHotDataGen(
            input=[PATH],
            batch_size=1,
            tilenumberx=i, 
            shuffle_permutations=True, 
            one_hot_encoding=True,
            Permutations_dictionary=perm_dict_1)
        
        perm_dict_2 = datagenerator.get_perm_dict()
        print(perm_dict_1 == perm_dict_2)
        datagenerator.test_data_gen()
        X,y = datagenerator.next()
        print(y)
        created_label = datagenerator.get_perm_from_label(tuple(y[0]))
        print(created_label)
        showPermImg(X[0], created_label, i)
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

    perm_img, label = getStitchedPermutation(img_data1, perm, PermutationDict, tilenumberx=number_tiles_x)
    plt.imshow(perm_img/255)
    plt.show()



def test():
    PATH = Path('../data_test/plantvillage/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG') 
    img1 = load_img(PATH, target_size=(255,255))
    img_data1 = img_to_array(img1, dtype = int)
    perm = (7, 4, 3, 2, 8, 5, 1, 6, 0)

    perm_img, label = getPermutation(
        img_data1,
        perm
        )
    showPermImg(perm_img/255, perm)
    plt.show()

    perm_img, label = getStitchedPermutation_four_tiles(
        img_data1,
        perm
        )
    plt.imshow(perm_img/255)
    plt.title("get stitched old")
    plt.show()

    perm_img, label = getStitchedPermutation(
        img_data1,
        perm
        )

    print(perm_img.shape)
    plt.title("get stitched new")
    print(label)
    plt.imshow(perm_img/255)
    plt.show()


if __name__=='__main__':
    import os 
    from pathlib import Path
    test()
    # main()