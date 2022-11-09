from BasePermutationDataGenerator import BasePermutationDataGenerator
from helper import *
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array


# always one hot encoded
class OneHotPermutationDataGenerator(BasePermutationDataGenerator):
    def __init__(self, 
        input, batch_size=64,
        reuse = 1, tile_number_x = 3,
        max_perms=25, target_size=(255,255,3),
        permutation_dict=None,
        stitched=False,
        shuffle=True, 
        shuffle_permutations=True, 
        preprocess_func=divide_by_255, 
        ):

        super(OneHotPermutationDataGenerator, self).__init__(
            input, batch_size=batch_size,
            reuse = reuse, tile_number_x = tile_number_x,
            max_perms=max_perms, target_size=target_size,
            permutation_dict=permutation_dict,
            stitched=stitched,
            shuffle=shuffle, 
            shuffle_permutations=shuffle_permutations,
            preprocess_func=preprocess_func
        )

    
    def get_batches(self, index_array):

        # features and labels 
        batch_x = np.zeros(
            (len(index_array), ) + self.output_size, 
            dtype='float32'
            )

        batch_y = np.zeros(
            (len(index_array),
            self.number_of_different_perms),
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
            X, _ = self.get_permuted_image(
                image, 
                perm, 
                None, 
                tilenumberx=self.tile_number_x)

            batch_x[i] = X
            batch_y[i] = self.perm_dict[tuple(perm)]       

        return batch_x, batch_y 


    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self.get_batches(index_array)

    def __str__(self) -> str:
        return super().__str__() + (
            f"# One-hot encoded\n"
        )

def test_datagen_stitched(datagen):
    print(datagen)
    X, y = datagen.next()
    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    random_int = random.randint(0, X.shape[0]-1)
    print(f"Permutation: {datagen.get_perm_from_label(y[random_int])}")
    print(f"Label: {y[random_int]}")
    plt.imshow(X[random_int])
    plt.show()

def test_datagen_tiled(datagen):
    print(datagen)
    X, y = datagen.next()
    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    random_int = random.randint(0, X.shape[0]-1)
    print(f"Permutation: {datagen.get_perm_from_label(y[random_int])}")
    print(f"Label: {y[random_int]}")
    showPermImg(X[random_int], datagen.get_perm_from_label(y[random_int]))
    plt.show()
    

def main():
    import os 
    from pathlib import Path
    import matplotlib.pyplot as plt

    PATH = Path(
        '../data_test/plantvillage/Apple___Apple_scab/'
        '0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG'
        ) 
    
    booleans = [True, False]
    number_of_tiles_for_tests = [3]
    
    for i in number_of_tiles_for_tests:
        # test stitched images
        for truth in booleans:
            datagen = OneHotPermutationDataGenerator(
                input=[PATH], 
                stitched=True,
                batch_size=1,
                shuffle_permutations=truth,
                tile_number_x=i
                )
            test_datagen_stitched(datagen)
        
        # test tiled images
        for truth in booleans:
            datagen = OneHotPermutationDataGenerator(
                input=[PATH], 
                stitched=False,
                batch_size=1,
                shuffle_permutations=truth,
                tile_number_x=i
                )
            test_datagen_tiled(datagen)


if __name__=='__main__':
    main()