

def train_model(image_set_A, image_set_B, training_parameters):
    '''
    A function that takes 2 sets as images and training parameters as input.

    crop A and B
    set A: is used for the classification
    set B: is used for the self-supervised learning

    Step 1: calculate baseline: 
    split set A into training, validation and test set
    use the frozen convolutional base in order to train a decoder
    test this decoder with the test set A
    In output: 
    number of images, number of classes, loss ,accuracy

    Step 2: training a decoder for the jigsaw problem:
    train jigsaw problem with set B

    Step 3: unfreeze the encoder and finetune it
    further train jigsaw problem with set B

    Step 4: calculate new model:
    split set A into training, validation and test set
    use the updated encoder in order to train a decoder
    and test the results on the test set.


    '''