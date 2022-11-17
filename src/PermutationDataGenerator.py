import numpy as np

from keras.preprocessing.image import Iterator
from tensorflow.keras.utils import load_img, img_to_array

from helper import * 
from cropping_tools import CropTool


class PermutationDataGenerator(Iterator):
    def __init__(
        self, input, batch_size=64,
        reuse = 1, tile_number_x = 3,
        max_perms=25, target_size=(255,255,3),
        load_size=None,
        permutation_dict=None,
        stitched=False,
        shuffle=True,  
        shuffle_permutations=True,
        preprocess_func=divide_by_255,
        one_hot_encoding=True,
        crop=False
        ):
        
        # metadata of the ML process
        self.batch_size = batch_size
        
        # metadata about the image
        self.target_size = target_size
        self.size_of_image = load_size or target_size
        self.tile_number_x = tile_number_x
        self.number_of_tiles = tile_number_x**2

        # loading the images
        if isinstance(input, list):
            self.im_as_files = True
            self.input_shape = (
                self.number_of_tiles, 
                self.size_of_image[0]//tile_number_x, # height dimension
                self.size_of_image[1]//tile_number_x, # width dimension
                self.size_of_image[2]                 # number of channels
                )

            self.images = input * reuse
        else:
            self.input_shape = self.images.shape[1:]
            self.images = input
        
        # add dimension if the images are greyscale
        if len(self.input_shape) == 2:
            self.input_shape = self.input_shape + (1,)

        self.number_of_images = len(self.images)


        self.stitched = stitched
        self.crop = crop
        self.get_permuted_image = getPermutation

        self.output_size = self.input_shape
        

        if self.stitched:
            self.get_permuted_image = getStitchedPermutation
            self.output_size = self.size_of_image

        if self.crop:
            self.get_permuted_image = CropTool(self.target_size)
            self.output_size = self.target_size

        # metadata about the permutations
        self.number_of_different_perms = min(
            max_perms, 
            np.math.factorial(self.number_of_tiles)
            )

        self.perm_dict = permutation_dict
        if not self.perm_dict:
            self.perm_dict = PermMapToOneHot(
                self.number_of_tiles,
                self.number_of_different_perms
                )
            self.reverse_dict = {
                tuple(val):key for (key, val) in self.perm_dict.items()
                }
        

        self.label_shape = self.number_of_tiles
        self.get_label_from_permutation = self.permutation_is_label
        self.one_hot_encoded = False

        if one_hot_encoding:
            self.label_shape = self.number_of_different_perms
            self.get_label_from_permutation = self.OHE_permutation
            self.one_hot_encoded = True

        self.shuffle_permutations = shuffle_permutations
        self.preprocessing_function = preprocess_func
        self.get_perm = self.get_random_perm

        if not self.shuffle_permutations:
            self.perms_labels = random.choices(
                list(self.perm_dict.items()),
                k=self.number_of_images) # with replacement
            self.perms = [
                perm_label[0] for perm_label in self.perms_labels
                ] # actual permutation
            self.labels = [
                perm_label[1] for perm_label in self.perms_labels
                ] # corresponding label 
            self.get_perm = self.get_existing_perm

        super(PermutationDataGenerator, self).__init__(
            self.number_of_images, batch_size, shuffle, None)

    def get_existing_perm(self, position):
        return self.perms[position]
    
    def get_random_perm(self, n):
        return random.choice(list(self.perm_dict.keys()))

    def get_perm_dict(self):
        return self.perm_dict

    def get_perm_from_label(self, label):
        if not isinstance(label, tuple):
            label = tuple(label)
        return self.reverse_dict[label]
    
    def permutation_is_label(self, perm):
        return perm 
    
    def OHE_permutation(self, perm):
        return self.perm_dict[perm]


    def _get_batches_of_transformed_samples(self, index_array):

        # features and labels 
        batch_x = np.zeros(
            (len(index_array), ) + self.output_size, 
            dtype='float32'
            )

        batch_y = np.zeros(
            (len(index_array),
            self.label_shape),
            dtype='float32')

        # iterate through the current batch
        for i, j in enumerate(index_array):
            if self.im_as_files:
                image = img_to_array(
                    load_img(
                        self.images[j], 
                        target_size=self.size_of_image
                        )
                    ) 
            else:
                image = self.images[j].squeeze()
            
            image = self.preprocessing_function(image)
            perm = self.get_perm(j)
            X, _ = self.get_permuted_image(image, perm, None, tilenumberx=self.tile_number_x)
            #X, _ = getStitchCropPermutation(image, perm, None, tilenumberx=self.tile_number_x, target_size=self.target_size)

            batch_x[i] = X
            batch_y[i] =  self.get_label_from_permutation(perm)

        return batch_x, batch_y 


    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

    
    def __str__(self) -> str:
        return super().__str__() + \
        (
            f"\n============================================================\n"
            f"\n# Tiles: {self.number_of_tiles}\n"
            f"# Permutations: {self.number_of_different_perms}\n"
            f"# Images: {self.number_of_images}\n"
            f"# Stitched: {self.stitched}\n"
            f"# Output shape: {self.output_size}\n"
            f"# get permutation from: {self.get_permuted_image}\n"
            f"# Preprocessing: {self.preprocessing_function}\n"
            f"# Shuffle Permutations: {self.shuffle_permutations}\n"
            f"# Encoding type: {self.get_label_from_permutation}\n"
        )
                

def test_datagen_stitched(datagen):
    print(datagen)
    X, y = datagen.next()
    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    random_int = random.randint(0, X.shape[0]-1)
    print(f"Label: {y[random_int]}")
    if datagen.one_hot_encoded:
        print(f"Permutation: {datagen.get_perm_from_label(y[random_int])}")
    plt.imshow(X[random_int])
    plt.show()

def test_datagen_tiled(datagen):
    print(datagen)
    X, y = datagen.next()
    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    random_int = random.randint(0, X.shape[0]-1)
    print(f"Label: {y[random_int]}")
    if datagen.one_hot_encoded:
        print(f"Permutation: {datagen.get_perm_from_label(y[random_int])}")
        showPermImg(X[random_int], datagen.get_perm_from_label(y[random_int]))
    else:
        showPermImg(X[random_int], y[random_int])
    plt.show()
        
def main():
    import os 
    from pathlib import Path
    import matplotlib as plt

    PATH = Path(
        '../data_test/plantvillage/Apple___Apple_scab/'
        '0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG'
        ) 
    
    booleans = [True, False]
    number_of_tiles_for_tests = [3]
    

    for i in number_of_tiles_for_tests:
        # test encoding types
        for encoding in booleans:
            # test stitched images
                for truth in booleans:
                    datagen = PermutationDataGenerator(
                        input=[PATH], 
                        stitched=True,
                        batch_size=1,
                        shuffle_permutations=truth,
                        tile_number_x=i,
                        one_hot_encoding=encoding
                        )
                    test_datagen_stitched(datagen)

        # test encoding types
        for encoding in booleans:
            # test tiled images
                for truth in booleans:
                    datagen = PermutationDataGenerator(
                        input=[PATH], 
                        stitched=False,
                        batch_size=1,
                        shuffle_permutations=truth,
                        tile_number_x=i,
                        one_hot_encoding=encoding
                        )
                    test_datagen_tiled(datagen)
    
if __name__=='__main__':
    main()
        