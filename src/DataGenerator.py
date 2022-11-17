from keras.preprocessing.image import Iterator
import numpy as np
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from helper import divide_by_255

class DataGenerator(Iterator):
    def __init__(
        self,
        input,
        labels, 
        preprocess=divide_by_255,
        reuse=1,
        batch_size=64, 
        target_size=(255,255,3),
        shuffle=True
        ):

        self.output_size = target_size
        self.number_of_classes = labels.shape[1]
        self.preprocessing_function = preprocess
        self.labels = labels
        self.target_size = target_size

        if isinstance(input, list):
            self.im_as_files = True
            self.images = input * reuse
            self.input_shape = target_size
        else:
            self.images = input
            self.input_shape = self.images.shape[1:]
        
        # add dimension if the images are greyscale
        if len(self.input_shape) == 2:
            self.input_shape = self.input_shape + (1,)

        self.number_of_images = len(self.images)

        super(DataGenerator, self).__init__(
            self.number_of_images, batch_size, shuffle, None)
    
    def _get_batches_of_transformed_samples(self, index_array):
        # features and labels 
        batch_x = np.zeros(
            (len(index_array), ) + self.output_size, 
            dtype='float32'
            )

        batch_y = np.zeros(
            (len(index_array),
            self.number_of_classes),
            dtype='float32')

        # iterate through the current batch
        for i, j in enumerate(index_array):
            if self.im_as_files:
                image = img_to_array(
                    load_img(
                        self.images[j], 
                        target_size=self.target_size
                        )
                    ) 
            else:
                image = self.images[j].squeeze()
            
            image = self.preprocessing_function(image)

            batch_x[i] = image
            batch_y[i] = self.labels[j]

        return batch_x, batch_y 

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self.get_batches(index_array)


def main():
    import os 
    from pathlib import Path
    import matplotlib.pyplot as plt

    PATH = Path(
        '../data_test/plantvillage/Apple___Apple_scab/'
        '0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG'
        ) 
    datagen = DataGenerator(
        input=[PATH],
        labels=np.array([[0,],]),
        batch_size=1
    )
    im, label = datagen.next()
    plt.imshow(im[0])
    plt.show()
    print(label[0])

if __name__=='__main__':
    main()