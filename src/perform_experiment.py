import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D, Concatenate, Lambda
from keras.models import Model
from keras.utils import to_categorical
from keras_preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

from cropping_tools import random_crop
from DataGenerator import DataGenerator
from PermutationDataGenerator import PermutationDataGenerator
from model_tools import *

def copyModel2Model(
    model_source, model_target, 
    number_of_layers_target=-1, 
    number_of_layers_source=-1):        
    # https://github.com/klaverhenrik/Deep-Learing-for-Visual-Recognition-2022 modified
    for l_tg,l_sr in zip(
        model_target.layers[:number_of_layers_target],
        model_source.layers[:number_of_layers_source]
        ):
        wk0 = l_sr.get_weights()
        l_tg.set_weights(wk0)
    print("model source was copied into model target")

def get_model_jigsaw(
    conv_base, 
    target_size: tuple,
    one_hot_encoding: bool, 
    stitched: bool, 
    number_of_permutations: int=None,
    number_of_tiles: int=None,
    tile_size: tuple=None):

    assert (one_hot_encoding and number_of_permutations) or number_of_tiles, \
        "Either one-hot encoding with number of perms or number of tiles is required"
    

    if stitched:
        tiles = Input(target_size)
        shared_conv = conv_base 
        concatonation = shared_conv(tiles)
        concatonation = GlobalAveragePooling2D()(concatonation)

        out = Dense(
            (1024+number_of_permutations)//2, 
            activation="relu", 
            kernel_initializer='he_normal'
            )(concatonation)

        if one_hot_encoding:
            out = Dense(
                number_of_permutations, 
                activation="softmax", 
                kernel_initializer='he_normal'
                )(out)
        else:
            out = Dense(number_of_tiles, kernel_initializer='he_normal')(out)

        out = Flatten()(out)
        model = Model(inputs=tiles, outputs=out)
    else:


        inputs = {}
        layers = {}
        embedds = {}

        shared_conv = conv_base 

        for i in range(number_of_tiles):
            inputs[f'tiles{i}'] = Input((tile_size,tile_size,3))
            layers[f'tile{i}'] = Lambda(lambda x: x[:,i,:,:,:])(tiles)
            layers[f'deep_layers{i}'] = shared_conv(inputs[f'tiles{i}'])
            # layers[f'deep_layers{i}'] = shared_conv(layers[f'tile{i}'])
            embedds[f'embedd{i}'] = Flatten()(layers[f'deep_layers{i}'])
        concatonation = Concatenate(axis=1)(list(embedds.values()))

        concatonation = shared_conv(tiles)
        concatonation = GlobalAveragePooling2D()(concatonation)

        out = Dense(
            (1024+number_of_permutations)//2, 
            activation="relu", 
            kernel_initializer='he_normal'
            )(concatonation)

        if one_hot_encoding:
            out = Dense(
                number_of_permutations, 
                activation="softmax", 
                kernel_initializer='he_normal')(out)
        else:
            out = Dense(number_of_tiles, kernel_initializer='he_normal')(out)
        out = Flatten()(out)

        # model = Model(inputs=tiles, outputs=out)
        model = Model(inputs=list(inputs.values()), outputs=out)

    return model


def perform_experiment(
    image_set_A, # list of filenames 
    set_A_labels, # one-hot encoding of labels of set A
    image_set_B,  # list of filenames
    batch_size=32,
    epochs_classification=10, 
    epochs_jigsaw=10,
    epochs_fine_tuning=10,
    reuse=1, 
    tile_number_x=3,
    number_of_permutations=25, 
    target_size=(255,255,3),
    stitched=True,
    shuffle_permutations=True,
    one_hot_encoding=True,
    preprocessing_func=preprocess_input,
    test_mode=False,
    ):
    '''how_to_perform_experiment.md'''
    if test_mode:
        epochs_classification = 2
        epochs_jigsaw = 2
        epochs_fine_tuning = 2
        reuse = 1
    
    assert stitched, "Non-stitched option was not tested" # delete as soon as it was tested
    


    # general parametes
    conv_base = MobileNet(
        weights='imagenet',
        include_top=False,
        input_shape=target_size)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name='Adam')

    # TODO: cropping of the images 


    # splitting set A
    num_classes = set_A_labels.shape[1]

    x_train, x_test, y_train, y_test = train_test_split(
        image_set_A, 
        set_A_labels, 
        test_size=0.2)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.1)
    
    training_generator_baseline = DataGenerator(
        x_train,
        y_train,
        preprocess=preprocess_input,
        shuffle=True)
    validation_generator_baseline = DataGenerator(
        x_val,
        y_val,
        preprocess=preprocess_input,
        shuffle=False)
    test_generator_baseline = DataGenerator(
        x_test,
        y_test,
        preprocess=preprocess_input,
        shuffle=False)



    # Step 1: Baseline model

    print("\n\n\n STARTING STAGE 1 \n\n\n")
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
        # print(layer)
        layer.trainable=True
    

    model_baseline.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

    history_baseline = model_baseline.fit(
        training_generator_baseline,
        batch_size=batch_size,
        epochs=epochs_classification,
        verbose=1,
        validation_data=validation_generator_baseline)

    score_baseline = model_baseline.evaluate(test_generator_baseline, verbose=0)
    loss_baseline = score_baseline[0]
    accuracy_baseline = score_baseline[1]
    # print(f'Test loss of the baseline model: {loss_baseline}')
    # print(f'Test accuracy of the baseline model: {accuracy_baseline}')

    # Step 2: training a decoder for the jigsaw problem
    print("\n\n\n STARTING STAGE 2 \n\n\n")

    model_jigsaw = get_model_jigsaw(
        conv_base=conv_base,
        target_size=target_size,
        one_hot_encoding=one_hot_encoding,
        number_of_permutations=number_of_permutations,
        stitched=stitched)

    for layer in model_jigsaw.layers:
        layer.trainable=False
    for layer in model_jigsaw.layers[-3:]: # TODO: please double check
        layer.trainable=True

    jigsaw_train, jigsaw_test = train_test_split(image_set_B)
    
    jigsaw_train_datagen = PermutationDataGenerator(
        input=jigsaw_train,
        batch_size=batch_size,
        reuse=reuse,
        tile_number_x=tile_number_x,
        max_perms=number_of_permutations,
        target_size=target_size,
        shuffle_permutations=shuffle_permutations,
        stitched=stitched,
        one_hot_encoding=one_hot_encoding,
        preprocess_func=preprocessing_func,
        permutation_dict=None,
        shuffle=True)

    jigsaw_test_datagen = PermutationDataGenerator(
        input=jigsaw_test,
        batch_size=batch_size,
        reuse=reuse,
        tile_number_x=tile_number_x,
        max_perms=number_of_permutations,
        target_size=target_size,
        shuffle_permutations=shuffle_permutations,
        stitched=stitched,
        one_hot_encoding=one_hot_encoding,
        preprocess_func=preprocessing_func,
        permutation_dict=jigsaw_train_datagen.get_perm_dict(),
        shuffle=False)

    if one_hot_encoding:
        model_jigsaw.compile(optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])
    else:
        model_jigsaw.compile(
            optimizer=optimizer,
            loss=RankingLoss(),
            metrics=[ProjectedRanksAccuracy(), PartialRanksAccuracy()])

    history_jigsaw = model_jigsaw.fit(
        jigsaw_train_datagen,
        epochs=epochs_jigsaw,
        validation_data=jigsaw_test_datagen,
        verbose=1)


    # Step 3: Fine-tuning
    print("\n\n\n STARTING STAGE 3 \n\n\n")
    model_fine_tuning = get_model_jigsaw(
        conv_base=conv_base,
        target_size=target_size,
        one_hot_encoding=one_hot_encoding,
        number_of_permutations=number_of_permutations,
        stitched=stitched)

    # copy parameters inplace TODO: do we have to do this after compiling?
    copyModel2Model(
        model_source=model_jigsaw,
        model_target=model_fine_tuning)
    
    # to be sure
    for layer in model_fine_tuning.layers:
        layer.trainable=True

    if one_hot_encoding:
        model_fine_tuning.compile(optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])
    else:
        model_fine_tuning.compile(
            optimizer=optimizer,
            loss=RankingLoss(),
            metrics=[ProjectedRanksAccuracy(), PartialRanksAccuracy()])

    history_fine_tuning = model_fine_tuning.fit(
        jigsaw_train_datagen,
        epochs=epochs_fine_tuning,
        validation_data=jigsaw_test_datagen,
        verbose=1)
    

    # Step 4: Classify again
    print("\n\n\n STARTING STAGE 4 \n\n\n")

    inputs_improved = Input(shape=target_size)
    x = conv_base(inputs_baseline)
    encoded = Flatten()(x)

    # decoder
    x = Dense(1024)(encoded)
    prediction = Dense(num_classes, activation="softmax")(x)

    model_improved = Model(inputs=inputs_baseline, outputs=prediction)

    copyModel2Model(
        model_source=model_jigsaw,
        model_target=model_improved,
        number_of_layers_source=-3,
        number_of_layers_target=-2     #TODO: double check 
    )

    model_improved.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

    history_improved = model_improved.fit(
        training_generator_baseline, # should be correct
        batch_size=batch_size,
        epochs=epochs_classification,
        verbose=1,
        validation_data=validation_generator_baseline) # should be correct

    score_improved = model_baseline.evaluate(test_generator_baseline, verbose=0) # should be correct

    return (
        history_baseline, 
        history_jigsaw, 
        history_fine_tuning, 
        history_improved, 
        score_baseline, 
        score_improved
    )


def main():
    import os
    import platform
    from pathlib import Path

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import cv2
    import numpy as np

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
    image_set_A = file_list[:100]
    image_set_B = file_list[100:200]


    labels = to_categorical(labels)
    labels_A = labels[:100,:]
    (
        history_baseline, 
        history_jigsaw, 
        history_fine_tuning, 
        history_improved, 
        score_baseline, 
        score_improved 
    ) = perform_experiment(
        image_set_A=image_set_A, 
        image_set_B=image_set_B, 
        set_A_labels=labels_A, 
        test_mode=True)


    


if __name__=='__main__':
    main()