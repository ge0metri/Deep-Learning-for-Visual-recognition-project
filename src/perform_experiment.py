import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2

from pathlib import Path
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

from PermutationDataGenerator import PermutationDataGenerator
from cropping_tools import random_crop
from DataGenerator import DataGenerator

def perform_experiment(
    image_set_A,
    set_A_labels,
    image_set_B, 
    batch_size=64,
    epochs_classification=20,
    reuse=1, 
    tile_number_x = 3,
    max_perms=25, 
    target_size=(255,255,3),
    stitched=True,
    shuffle_permutations=True,
    one_hot_encoding=True
    ):
    '''how_to_perform_experiment.md'''

    # general parametes
    conv_base = MobileNet(
        weights='imagenet',
        include_top=False,
        input_shape=target_size
        )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name='Adam'
        )

    # TODO: cropping of the images has to be done before using this function!!


    # splitting set A
    num_classes = set_A_labels.shape[1]

    x_train, x_test, y_train, y_test = train_test_split(
        image_set_A, 
        set_A_labels, 
        test_size=0.2
        )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.1
        )
    
    training_generator_baseline = DataGenerator(
        x_train,
        y_train,
        preprocess=preprocess_input,
        shuffle=True
    )
    validation_generator_baseline = DataGenerator(
        x_val,
        y_val,
        preprocess=preprocess_input,
        shuffle=False
    )
    test_generator_baseline = DataGenerator(
        x_test,
        y_test,
        preprocess=preprocess_input,
        shuffle=False
    )



    # Step 1: Baseline model

    # encoder
    inputs_baseline = Input(shape=target_size)
    x = conv_base(inputs_baseline)
    encoded = Flatten()(x)

    # decoder
    x = Dense(1024)(encoded)
    prediction = Dense(num_classes, activation="softmax")(x)

    model_baseline = Model(inputs=inputs_baseline, outputs=prediction)


    for layer in model_baseline.layers:
        layer.trainable=False
    for layer in model_baseline.layers[-2:]:
        print(layer)
        layer.trainable=True
    

    model_baseline.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

    history = model_baseline.fit(
        training_generator_baseline,
        batch_size=batch_size,
        epochs=epochs_classification,
        verbose=1,
        validation_data=validation_generator_baseline)

    score = model_baseline.evaluate(test_generator_baseline, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def main():
    import platform
    import cv2
    import numpy as np
    import os
    from pathlib import Path

    p = Path(
        "../data_test/plantvillage"
        )
    classes = [
        "all"
        ]

    if "all" in classes:
        classes = os.listdir(p)

    file_list = []
    labels = []
    for label, c in enumerate(classes):
        print(c,end=" ")
        current_file_list = [x for x in (p/c).iterdir() if x.is_file()]
        for f in current_file_list:
            img = cv2.imread(str(f))
            if img is None:
                print(f'Failed to open {f}. Deleting file')
                os.remove(str(f))
        print(len(current_file_list))
        file_list += current_file_list
        labels += [label] * len(current_file_list)

    print(len(file_list))
    print(len(labels))
    image_set_A = file_list
    image_set_B = file_list

    labels = to_categorical(labels)
    perform_experiment(image_set_A=image_set_A, image_set_B=image_set_B, set_A_labels=labels)
    


if __name__=='__main__':
    main()