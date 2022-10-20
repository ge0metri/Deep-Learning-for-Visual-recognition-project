import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from itertools import permutations


def PermMapToOneHot(num):
    perms = list(permutations(range(num)))
    fact = np.math.factorial(num)
    eye = np.eye(fact)
    return {perms[i]:eye[i] for i in range(fact)}

def main():
    import os 
    img1 = load_img(os.path.join(os.getcwd(), 'data_test/plantvillage/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG'), target_size=(255, 255))
    img_data1 = img_to_array(img1, dtype = int)
    showPermImg(*getPermutation(img_data1, 2), 2)
    
    # PermDict = PermMapToOneHot(4)
    # ReverseDict = {tuple(val):key for (key, val) in PermDict.items()}
    # dataGen = PermNetDataGenerator(['data_test/plantvillage/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG'],1,tilenumberx=2)
    # X,y = dataGen.next()
    # print(y)
    # showPermImg(X[0], ReverseDict[tuple(y[0])], 2)
    plt.show()


    

def getPermutation(image_array, tilenumberx=3, shuffle = True, rules = False):
    """Takes an image as an array, and returns a permuted image as an array
    with corresponding labels.
    """
    idx = range(tilenumberx**2)
    if shuffle:
        idx = np.random.permutation(tilenumberx**2)
    tilesize_h = image_array.shape[0]//(tilenumberx)
    tilesize_w = image_array.shape[1]//(tilenumberx)

    tiles = [image_array[(i//tilenumberx)*tilesize_h:(i//tilenumberx+1)*tilesize_h,(i%tilenumberx)*tilesize_w:(i%tilenumberx+1)*tilesize_w,:] for i in idx]
    out = np.array(tiles)
    if rules:
        idx = rules[tuple(idx)]

    return out, idx


def showPermImg(X, y):
    tilenum = len(X)
    plt.figure(figsize=(tilenum,tilenum))
    for i in range(tilenum**2):
        plt.subplot(tilenum,tilenum,i+1)
        plt.imshow(X[i])
        plt.xticks([]), plt.yticks([])
        plt.title(int(y[i]))

from keras.preprocessing.image import Iterator
class PermNetDataGenerator(Iterator):

    def __init__(self, input, batch_size=64,
                 preprocess_func=None, shuffle=False, reuse = 1, tilenumberx = 3):
        if type(input) == list:
            self.im_as_files = True
            self.input_shape = (tilenumberx**2,255//tilenumberx,255//tilenumberx,3)
            self.images = input*reuse
        else:
            self.input_shape = self.images.shape[1:]
            self.images = input
        self.tilenumberx = tilenumberx
        self.batch_size = batch_size
        self.preprocess_func = preprocess_func
        self.shuffle = shuffle

        # add dimension if the images are greyscale
        if len(self.input_shape) == 2:
            self.input_shape = self.input_shape + (1,)
        N = len(self.images)

        super(PermNetDataGenerator, self).__init__(N, batch_size, shuffle, None)
        
    def _get_batches_of_transformed_samples(self, index_array):
        # create array to hold the images
        batch_x = np.zeros((len(index_array),) + self.input_shape, dtype='float32')
        # create array to hold the labels
        batch_y = np.zeros((len(index_array),self.tilenumberx**2), dtype='float32')

        # iterate through the current batch
        for i, j in enumerate(index_array):
            
            if self.im_as_files:
                image = img_to_array(load_img(self.images[j], target_size=(255, 255))) / 255 #should prob not be hardcoded
            else:
                image = self.images[j].squeeze()
            X, y = getPermutation(image, tilenumberx=self.tilenumberx)
            # store the image and label in their corresponding batches
            batch_x[i] = X
            batch_y[i] = y

        # preprocess input images
        if self.preprocess_func:
            tiles = list(range(self.tilenumberx**2))
            for i in tiles:
                batch_x[:,i:,:,:,:] = self.preprocess_func(batch_x[:,i:,:,:,:].squeeze())

        return batch_x, batch_y

    def next(self):
        with self.lock:
            # get input data index and size of the current batch
            index_array = next(self.index_generator)
        # create array to hold the images
        return self._get_batches_of_transformed_samples(index_array)



if __name__=='__main__':
    main()