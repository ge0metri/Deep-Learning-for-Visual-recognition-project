import numpy as np 
from tensorflow import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
from itertools import permutations
import random

def divide_by_255(image):
    return image/255

def showPermImg(X, perm = None, tilenum = 3):
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

def getStitchedPermutationConcat(image_as_array, perm: tuple, label_from_perm=None, tilenumberx=3):
    """
    Takes an image as an array and a corresponding permutations
    that agrees with tilenumberx, and returns a permuted image as an array
    """

    idx = perm
    tilesize_h = image_as_array.shape[0]//(tilenumberx)
    tilesize_w = image_as_array.shape[1]//(tilenumberx)
    tiles = np.zeros(image_as_array.shape)

    tiles = [
        image_as_array[
            (i//tilenumberx)*tilesize_h:(i//tilenumberx+1)*tilesize_h, # cutting x dim
            (i%tilenumberx)*tilesize_w:(i%tilenumberx+1)*tilesize_w,   # cutting y dim
            :                                                          # keep channels
            ]
            for i in idx
        ]
    
    rows = []
    i = 0
    while i < tilenumberx**2:
        row = np.concatenate(tiles[i:i+tilenumberx],axis=1)
        rows.append(row)
        i = i + tilenumberx
        
    rows = np.concatenate(rows[:], axis=0)   

    
    out = np.array(rows)

    if label_from_perm:
        label = label_from_perm[idx]
        return out, label 
    
    return out, perm

def getStitchedPermutation(image_as_array, perm: tuple, label_from_perm=None, tilenumberx=3):
    """
    Takes an image as an array and a corresponding permutations
    that agrees with tilenumberx, and returns a permuted image as an array
    """

    idx = perm
    tilesize_h = image_as_array.shape[0]//(tilenumberx)
    tilesize_w = image_as_array.shape[1]//(tilenumberx)
    tiles = np.zeros(image_as_array.shape)
    for i, r in enumerate(idx):
        tiles[
            (i//tilenumberx)*tilesize_h:(i//tilenumberx+1)*tilesize_h,
            (i%tilenumberx)*tilesize_w:(i%tilenumberx+1)*tilesize_w,
            :
            ] = image_as_array[
            (r//tilenumberx)*tilesize_h:(r//tilenumberx+1)*tilesize_h, # cutting x dim
            (r%tilenumberx)*tilesize_w:(r%tilenumberx+1)*tilesize_w,   # cutting y dim
            :                                                          # keep channels
            ]
    out = np.array(tiles)

    if label_from_perm:
        label = label_from_perm[idx]
        return out, label 
    
    return out, perm

class PlotLearning(tf.keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        metrics = [x for x in logs if 'val' not in x]
        
        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2), 
                        self.metrics[metric], 
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics['val_' + metric], 
                            label='val_' + metric)
                
            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()
